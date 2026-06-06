import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, model_config):#  num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device,
                 #method="OLS"):
        """
        Generalised VAR (GVAR) model based on self-explaining neural networks.

        @param num_vars: number of variables (p).
        @param order:  model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        @param method: fitting algorithm (currently, only "OLS" is supported).
        """
        super(Model, self).__init__()

        num_vars = model_config.enc_in
        order = model_config.seq_len
        hidden_layer_size = model_config.d_model
        num_hidden_layers = model_config.e_layers
        device = model_config.device
        method ="OLS"

        # Networks for amortising generalised coefficient matrices.
        self.coeff_nets = nn.ModuleList()

        # Instantiate coefficient networks
        for k in range(order):
            modules = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
            if num_hidden_layers > 1:
                for j in range(num_hidden_layers - 1):
                    modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
            modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars**2)))
            self.coeff_nets.append(nn.Sequential(*modules))

        # Some bookkeeping
        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers

        self.device = device

        self.method = method

    # Initialisation
    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    # Forward propagation,
    # returns predictions and generalised coefficients corresponding to each prediction
    def forward_old(self, inputs: torch.Tensor):
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")

        coeffs = None
        if self.method is "OLS":
            preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
            for k in range(self.order):
                coeff_net_k = self.coeff_nets[k]
                coeffs_k = coeff_net_k(inputs[:, k, :])
                coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_vars, self.num_vars))
                if coeffs is None:
                    coeffs = torch.unsqueeze(coeffs_k, 1)
                else:
                    coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)
                coeffs[:, k, :, :] = coeffs_k
                if self.method is "OLS":
                    preds += torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze()
        elif self.method is "BFT":
            NotImplementedError("Backfitting not implemented yet!")
        else:
            NotImplementedError("Unsupported fitting method!")

        return preds, coeffs
    def forward(self, inputs: torch.Tensor):
        """
        inputs: (B, T, F)
        returns:
            preds: (B, T, F)
            coeffs: (B, T, order, F, F)
        """

        B, T, F = inputs.shape
        device = inputs.device

        if F != self.num_vars:
            raise ValueError("Feature dimension mismatch")

        if self.method != "OLS":
            raise NotImplementedError("Only OLS is implemented")

        preds = torch.zeros(B, T, F, device=device)
        coeffs = torch.zeros(B, T, self.order, F, F, device=device)

        # only valid timesteps
        for t in range(self.order, T):

            pred_t = torch.zeros(B, F, device=device)

            for k in range(self.order):

                x_lag = inputs[:, t - k - 1, :]  # (B, F)

                coeff_net_k = self.coeff_nets[k]
                coeffs_k = coeff_net_k(x_lag).view(B, F, F)

                coeffs[:, t, k] = coeffs_k

                pred_t = pred_t + torch.matmul(
                    coeffs_k,
                    x_lag.unsqueeze(-1)
                ).squeeze(-1)

            preds[:, t] = pred_t

        return preds, coeffs