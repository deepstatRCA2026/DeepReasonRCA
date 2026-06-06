import torch.nn as nn
import torch

def compute_kl_divergence(us, device: torch.device):
    """
    Compute the KL divergence between the empirical distribution of the input samples
    and an isotropic standard Gaussian distribution using PyTorch.

    Parameters:
    samples (Tensor): A 2D tensor with rows as samples and columns as features.

    Returns:
    Tensor: The KL divergence between the empirical distribution of the samples
            and the standard Gaussian distribution.
    """

    # Calculate the empirical mean and covariance matrix of the samples
    mean_p = torch.mean(us, dim=0)
    cov_p = torch.cov(us.t())

    # Dimensionality of the distribution
    d = mean_p.shape[0]

    eigenvalues = torch.linalg.eigvalsh(cov_p)
    condition_number = eigenvalues.max() / eigenvalues.clamp(min=1e-9).min()
    regularization_term = condition_number * 1e-6
    cov_p += torch.eye(d, device=device) * regularization_term
    # Ensure the covariance matrix is full rank
    # cov_p += 1e-9 * torch.eye(d).to(device)

    # Compute the trace term
    trace_term = torch.trace(cov_p)

    # Compute the product of means term (since mean_q is zero, this is just mean_p squared)
    means_term = torch.dot(mean_p, mean_p)

    # # Compute the determinant term
    # log_det_cov_p = torch.logdet(cov_p)
    try:
        L = torch.linalg.cholesky(cov_p)
        log_det_cov_p = 2 * torch.log(torch.diagonal(L)).sum()
    except RuntimeError:
        # Handle the case where Cholesky decomposition fails
        log_det_cov_p = torch.logdet(cov_p)

    # Compute the KL divergence using the formula
    kl_div = means_term + trace_term - d + log_det_cov_p
    if torch.isnan(kl_div).any():
        print('nan')
        print(f'mean_p: {mean_p}')
        print(f'cov_p: {cov_p}')
        print(f'trace_term: {trace_term}')
        print(f'means_term: {means_term}')
        print(f'log_det_cov_p: {log_det_cov_p}')
        print(f'kl_div: {kl_div}')
        raise ValueError('KL divergence is NaN')


    return kl_div

def sliding_window_view_torch(x, window_size: int):
    """
    A function to create a 2D sliding window view of a 2D PyTorch tensor.

    Args:
    x (torch.Tensor): The input 2D tensor.
    window_size (int): Window size.

    Returns:
    torch.Tensor: A tensor with the sliding windows.
    """
    # Calculate output shape
    output_shape = (x.size(0) - window_size + 1, window_size, x.size(1))
    # Calculate strides
    strides = (x.stride(0), x.stride(0), x.stride(1))
    # Create a view
    return x.as_strided(size=output_shape, stride=strides)


class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device):
        """
        Generalised VAR (GVAR) model based on self-explaining neural networks.
        @param num_vars: number of variables (p).
        @param order:  model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        """
        super(SENNGC, self).__init__()

        # Networks for amortising generalised coefficient matrices.
        self.coeff_nets = nn.ModuleList()

        # Instantiate coefficient networks
        for k in range(order):
            modules = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
            if num_hidden_layers > 1:
                for j in range(num_hidden_layers - 1):
                    modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
            modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars**2), nn.Tanh()))
            self.coeff_nets.append(nn.Sequential(*modules))

        # Some bookkeeping
        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers
        self.device = device


    # Initialisation
    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    # Forward propagation,
    # returns predictions and generalised coefficients corresponding to each prediction
    def forward(self, inputs: torch.Tensor):
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")

        coeffs = None
        preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
        for k in range(self.order):
            coeff_net_k = self.coeff_nets[k]
            coeffs_k = coeff_net_k(inputs[:, k, :])
            coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_vars, self.num_vars))
            if coeffs is None:
                coeffs = torch.unsqueeze(coeffs_k, 1)
            else:
                coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)
            # coeffs[:, k, :, :] = coeffs_k
            preds = preds + torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze()
        return preds, coeffs
    
class Model(nn.Module):
    def __init__(self,model_config):# num_vars: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device,
                 #window_size: int):
        super(Model, self).__init__()

        self.num_vars = model_config.enc_in
        self.window_size = 3 # we choose a fixed window size of 3, as this is a good tradeoff for seq leng of 12
        self.hidden_layer_size = model_config.d_model
        self.num_hidden_layers = model_config.e_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = SENNGC(self.num_vars, self.window_size, self.hidden_layer_size, self.num_hidden_layers, self.device)
        self.decoder = SENNGC(self.num_vars, self.window_size, self.hidden_layer_size, self.num_hidden_layers, self.device)
        self.decoder_prev = SENNGC(self.num_vars, self.window_size, self.hidden_layer_size, self.num_hidden_layers, self.device)
 
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.decoder_prev.to(self.device)


    def encoding(self, xs):
        """
        xs: (B, T, N)
        Produces:
            us: (B*(T-K), N)
            encoder_coeffs
            nexts: (B*(T-K), N)
            winds: (B*(T-K), K, N)
        """

        xs = xs.to(self.device)
        B, T, N = xs.shape
        K = self.window_size

        # must allow K context + 1 prediction step
        assert T >= K + 1, f"T={T} must be >= window_size+1={K+1}"
        """example
        (64, 12, 47)
            → unfold
            → (576, 4, 47)
            → split
            → (576, 3, 47), (576, 47)
        """
        # (B, T-K, K+1, N)
        windows = xs.unfold(1, K + 1, 1)

        # flatten temporal batches
        windows = windows.reshape(-1, K + 1, N)

        winds = windows[:, :-1, :].float()   # (B*(T-K), K, N)
        nexts = windows[:, -1, :].float()    # (B*(T-K), N)

        preds, encoder_coeffs = self.encoder(winds)

        us = preds - nexts  # residual

        return us, encoder_coeffs, nexts, winds

    def decoding(self, us, winds, xs, add_u=True):
        """
        us: (T, N)
        winds: (T-K, K, N)
        xs: (T, N) raw series
        """

        K = self.window_size

        # -------------------------
        # u stream
        # -------------------------
        u_windows = sliding_window_view_torch(us, K + 1)
        u_winds = u_windows[:, :-1, :]
        u_next = u_windows[:, -1, :]

        u_pred, u_coeffs = self.decoder(u_winds)

        # -------------------------
        # x stream
        # -------------------------
        x_pred, x_coeffs = self.decoder_prev(winds)

        # -------------------------
        # ALIGN EVERYTHING TO SAME TIME BASE
        # -------------------------
        min_len = min(u_pred.shape[0], x_pred.shape[0], u_next.shape[0])

        u_pred = u_pred[:min_len]
        x_pred = x_pred[:min_len]
        u_next = u_next[:min_len]

        # IMPORTANT: align ground truth too
        x_target = xs[K:K + min_len]

        # -------------------------
        # reconstruction
        # -------------------------
        if add_u:
            x_hat = x_pred + u_pred + u_next
        else:
            x_hat = x_pred + u_pred

        return x_hat, x_target, u_coeffs, x_coeffs

    def forward(self, x, add_u=True):
        """
        PURE INTERFACE (FITS-style)
        """

        us, encoder_coeffs, nexts, winds = self.encoding(x)

        kl_div = compute_kl_divergence(us, self.device)

        nexts_hat,nexts, decoder_coeffs, prev_coeffs = self.decoding(us, winds, nexts, add_u)

        return (
            nexts_hat,
            nexts,
            encoder_coeffs,
            decoder_coeffs,
            prev_coeffs,
            kl_div,
            us
        )



# will be handled in the training loop in RMDnet
#def _sparsity_loss(self, coeffs, alpha):
#    norm2 = torch.mean(torch.norm(coeffs, dim=1, p=2))
#    norm1 = torch.mean(torch.norm(coeffs, dim=1, p=1))
#    return (1 - alpha) * norm2 + alpha * norm1
#
#def _smoothness_loss(self, coeffs):
#    return torch.norm(coeffs[:, 1:, :, :] - coeffs[:, :-1, :, :], dim=1).mean()
    
#def compute loss(self, x, add_u=True):
        #loss_recon = self.mse_loss(nexts_hat, nexts)
        #logging.info('Reconstruction loss: %s', loss_recon.item())
#
        #loss_encoder_coeffs = self._sparsity_loss(encoder_coeffs, self.encoder_alpha)
        #logging.info('Encoder coeffs loss: %s', loss_encoder_coeffs.item())
#
        #loss_decoder_coeffs = self._sparsity_loss(decoder_coeffs, self.decoder_alpha)
        #logging.info('Decoder coeffs loss: %s', loss_decoder_coeffs.item())
#
        #loss_prev_coeffs = self._sparsity_loss(prev_coeffs, self.decoder_alpha)
        #logging.info('Prev coeffs loss: %s', loss_prev_coeffs.item())
#
        #loss_encoder_smooth = self._smoothness_loss(encoder_coeffs)
        #logging.info('Encoder smooth loss: %s', loss_encoder_smooth.item())
#
        #loss_decoder_smooth = self._smoothness_loss(decoder_coeffs)
        #logging.info('Decoder smooth loss: %s', loss_decoder_smooth.item())
#
        #loss_prev_smooth = self._smoothness_loss(prev_coeffs)
        #logging.info('Prev smooth loss: %s', loss_prev_smooth.item())
#
        #loss_kl = kl_div
        #logging.info('KL loss: %s', loss_kl.item())
#
        #loss = (loss_recon +
        #        self.encoder_lambda * loss_encoder_coeffs +
        #        self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs) +
        #        self.encoder_gamma * loss_encoder_smooth +
        #        self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth) +
        #        self.beta * loss_kl)
        #logging.info('Total loss: %s', loss.item())