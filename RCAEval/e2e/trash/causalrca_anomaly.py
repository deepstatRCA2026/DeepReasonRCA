from csv import writer
import math
import time
import warnings
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import torch.optim as optim
from sknetwork.ranking import PageRank
from torch.optim import lr_scheduler
from RCAEval.io.time_series import preprocess, drop_constant
from RCAEval.e2e.models.fits import Model as FITSModel

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from RCAEval.io.time_series import preprocess

_EPS = 1e-10

# ===============================
# Pure reconstruction encoder
# ===============================
class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0.0, factor=True, tol=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_xdims, n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.dropout = nn.Dropout(do_prob)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        z = self.fc2(h)
        return z  # latent representation


# ===============================
# Pure reconstruction decoder
# ===============================
class MLPDecoder(nn.Module):
    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size, n_hid, do_prob=0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_in_z, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.dropout = nn.Dropout(do_prob[0])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = self.dropout(h)
        x_recon = self.fc2(h)
        return x_recon


# ========================================
# VAE utility functions
# ========================================
def get_triu_indices(num_nodes):  # NOTE
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):  # NOTE
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):  # NOTE
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):  # NOTE
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.0
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):  # NOTE
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.0
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def kl_gaussian_sem(preds):  # NOTE
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0))) * 0.5


def nll_gaussian(preds, target, variance, add_const=False):  # NOTE
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.0 * np.exp(2.0 * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


def preprocess_adj_new_old(adj):  # NOTE
    if CONFIG.cuda:
        adj_normalized = torch.eye(adj.shape[0]).double().cuda() - (adj.transpose(0, 1))
    else:
        adj_normalized = torch.eye(adj.shape[0]).double() - (adj.transpose(0, 1))
    return adj_normalized


def preprocess_adj_new1_old(adj):  # NOTE
    if CONFIG.cuda:
        adj_normalized = torch.inverse(
            torch.eye(adj.shape[0]).double().cuda() - adj.transpose(0, 1)
        )
    else:
        adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double() - adj.transpose(0, 1))
    return adj_normalized




def preprocess_adj_new(adj):
    """Compute I - A^T"""
    device = adj.device
    I = torch.eye(adj.shape[0], device=device, dtype=adj.dtype)  # Use same dtype as input
    return I - adj.transpose(0, 1)

def preprocess_adj_new1(adj):
    """Compute (I - A^T)^(-1)"""
    device = adj.device
    I = torch.eye(adj.shape[0], device=device, dtype=adj.dtype)
    return torch.linalg.inv(I - adj.transpose(0, 1))

def isnan(x):  # NOTE
    return x != x


def matrix_poly(matrix, d):  # NOTE
    if CONFIG.cuda:
        x = torch.eye(d).double().cuda() + torch.div(matrix, d)
    else:
        x = torch.eye(d).double() + torch.div(matrix, d)
    return torch.matrix_power(x, d)


# matrix loss: makes sure at least A connected to another parents for child
def A_connect_loss(A, tol, z):  # NOTE
    d = A.size()[0]
    loss = 0
    for i in range(d):
        loss += 2 * tol - torch.sum(torch.abs(A[:, i])) - torch.sum(torch.abs(A[i, :])) + z * z
    return loss


# element loss: make sure each A_ij > 0
def A_positive_loss(A, z_positive):  # NOTE
    result = -A + z_positive * z_positive
    loss = torch.sum(result)

    return loss


class CONFIG:  # NOTE
    """Dataclass with app parameters"""

    def __init__(self):
        pass

    # You must change this to the filename you wish to use as input data!
    # data_filename = "alarm.csv"

    # Epochs
    epochs = 10

    # Batch size (note: should be divisible by sample size, otherwise throw an error)
    batch_size = 50

    # Learning rate (baseline rate = 1e-3)
    lr = 1e-3

    x_dims = 1
    z_dims = 1
    # data_variable_size = 12
    optimizer = "Adam"
    graph_threshold = 0.3
    tau_A = 0.0
    lambda_A = 0.0
    c_A = 1
    use_A_connect_loss = 0
    use_A_positiver_loss = 0
    # no_cuda = True
    seed = 42
    encoder_hidden = 128
    decoder_hidden = 128
    temp = 0.5
    k_max_iter = 1
    encoder = "mlp"
    decoder = "mlp"
    no_factor = False
    encoder_dropout = 0.0
    decoder_dropout = (0.0,)
    h_tol = 1e-8
    lr_decay = 200
    gamma = 1.0
    prior = False


CONFIG.cuda = torch.cuda.is_available()
CONFIG.factor = not CONFIG.no_factor



import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from statsmodels.tsa.stattools import grangercausalitytests
from utils import (pot)

# =========================
# POT algorithm wrapper
# =========================
def compute_pot_scores(errors, risk=1e-2, init_level=0.98, num_candidates=10, epsilon=1e-8):
    """
    Compute POT-based anomaly scores from reconstruction errors.
    Includes safeguards for tiny values and fallback to quantile.
    """
    errors = np.asarray(errors).ravel()

    # Rescale if values are too small
    if errors.max() < 1e-6:
        errors = errors * 1e6

    try:
        # POT thresholding
        z, t = pot(errors,
                   q=risk,
                   level=init_level,
                   n_candidate=num_candidates,
                   epsilon=epsilon)
    except Exception:
        # Fallback: simple quantile threshold
        z = np.quantile(errors, 1 - risk)
        t = np.where(errors > z)[0]

    # Normalize scores
    scores = np.maximum(0, errors - z)
    if scores.max() > 0:
        scores = scores / scores.max()

    return scores, z, t


# =========================
# Main causal RCA function
# =========================
def causalrca_anomaly(data, inject_time=None, dataset=None, with_bg=False, **kwargs):
    if type(data) == dict: # multimodal 
        metric = data["metric"]
        logts = data["logts"]
        # traces_err = data["tracets_err"]
        # traces_lat = data["tracets_lat"]

        # === metric ===
        metric = metric.iloc[::15, :]

        # == metric ==
        normal_metric = metric[metric["time"] < inject_time]
        anomal_metric = metric[metric["time"] >= inject_time]
        normal_metric = preprocess(data=normal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
        anomal_metric = preprocess(data=anomal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
        intersect = [x for x in normal_metric.columns if x in anomal_metric.columns]
        normal_metric = normal_metric[intersect]
        anomal_metric = anomal_metric[intersect]
        metric = pd.concat([normal_metric, anomal_metric], axis=0, ignore_index=True)
        data = metric
        print(f"{normal_metric.shape=}")
        print(f"{anomal_metric.shape=}")
        print(f"{metric.shape=}")
        print("with metric", data.shape)

        # == logts ==
        logts = drop_constant(logts)
        normal_logts = logts[logts["time"] < inject_time].drop(columns=["time"])
        anomal_logts = logts[logts["time"] >= inject_time].drop(columns=["time"])
        log = pd.concat([normal_logts, anomal_logts], axis=0, ignore_index=True)
        data = pd.concat([data, log], axis=1)
        print(f"{normal_logts.shape=}")
        print(f"{anomal_logts.shape=}")
        print(f"{log.shape=}")
        print("with log", data.shape)
        data.to_csv("debug_withlog.csv", index=False)

        # print(f"{normalize=} {addup=}")

        # # == traces_err ==
        # if dataset == "mm-tt" or dataset == "mm-ob":
        #     traces_err = traces_err.fillna(method='ffill')
        #     traces_err = traces_err.fillna(0)
        #     traces_err = drop_constant(traces_err)

        #     normal_traces_err = traces_err[traces_err["time"] < inject_time].drop(columns=["time"])
        #     anomal_traces_err = traces_err[traces_err["time"] >= inject_time].drop(columns=["time"])
        #     trace = pd.concat([normal_traces_err, anomal_traces_err], axis=0, ignore_index=True)
        #     data = pd.concat([data, trace], axis=1)
        #     print(f"{normal_traces_err.shape=}")
        #     print(f"{anomal_traces_err.shape=}")
        #     print(f"{trace.shape=}")
        #     print("with traces_err", data.shape)
        # 
        #  # == traces_lat ==
        # if dataset == "mm-tt" or dataset == "mm-ob":
        #     traces_lat = traces_lat.fillna(method='ffill')
        #     traces_lat = traces_lat.fillna(0)
        #     traces_lat = drop_constant(traces_lat)
        #     normal_traces_lat = traces_lat[traces_lat["time"] < inject_time].drop(columns=["time"])
        #     anomal_traces_lat = traces_lat[traces_lat["time"] >= inject_time].drop(columns=["time"])
        #     trace = pd.concat([normal_traces_lat, anomal_traces_lat], axis=0, ignore_index=True)
        #     data = pd.concat([data, trace], axis=1)
        #     print(f"{normal_traces_lat.shape=}")
        #     print(f"{anomal_traces_lat.shape=}")
        #     print(f"{trace.shape=}")
        #     print("with traces_lat", data.shape)

        # dump to debug.csv
        # data.to_csv("debug.csv", index=False)
        # drop duplicated columns
        data = data.loc[:, ~data.columns.duplicated()]
        data = data.fillna(0)

    else:
        data = preprocess(
            data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
        )

    data /= data.max()

    data_sample_size = data.shape[0]
    data_variable_size = data.shape[1]

    node_names = data.columns.to_list()

    # graph construction, get the adj
    train_data = data

    # Generate off-diagonal interaction graph
    off_diag = np.ones([data_variable_size, data_variable_size]) - np.eye(data_variable_size)

    # add adjacency matrix A
    num_nodes = data_variable_size
    adj_A = np.zeros((num_nodes, num_nodes))

    encoder = MLPEncoder(
        data_variable_size * CONFIG.x_dims,
        CONFIG.x_dims,
        CONFIG.encoder_hidden,
        int(CONFIG.z_dims),
        adj_A,
        batch_size=CONFIG.batch_size,
        do_prob=CONFIG.encoder_dropout,
        factor=CONFIG.factor,
    ).double()

    decoder = MLPDecoder(
        data_variable_size * CONFIG.x_dims,
        CONFIG.z_dims,
        CONFIG.x_dims,
        encoder,
        data_variable_size=data_variable_size,
        batch_size=CONFIG.batch_size,
        n_hid=CONFIG.decoder_hidden,
        do_prob=CONFIG.decoder_dropout,
    ).double()

    # ===================================
    # set up training parameters
    # ===================================
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=CONFIG.lr_decay, gamma=CONFIG.gamma)

    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(data_variable_size)
    tril_indices = get_tril_offdiag_indices(data_variable_size)

    if CONFIG.cuda:
        encoder.cuda()
        decoder.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    # compute constraint h(A) value
    def _h_A(A, m):
        expm_A = matrix_poly(A * A, m)
        h_A = torch.trace(expm_A) - m
        return h_A

    prox_plus = torch.nn.Threshold(0.0, 0.0)

    def stau(w, tau):
        w1 = prox_plus(torch.abs(w) - tau)
        return torch.sign(w) * w1

    def update_optimizer(optimizer, original_lr, c_A):
        """related LR to c_A, whenever c_A gets big, reduce LR proportionally"""
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in optimizer.param_groups:
            parame_group["lr"] = lr

        return optimizer, lr

    # ----------------------------
    # Precompute diffusion schedule once (outside training loop)
    # ----------------------------
    T = 100  # max diffusion steps
    device = torch.device("cuda" if CONFIG.cuda else "cpu")

    betas = torch.linspace(1e-4, 0.02, T, device=device)   # put directly on correct device
    alpha = 1.0 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)   # [T]


    timing_stats = {"enc": [], "dec": [], "loss": [], "back": []}

    # ===================================
    # training:
    # ===================================
    def train_old(epoch, best_val_loss, lambda_A, c_A, optimizer):
        t = time.time()
        nll_train = []
        kl_train = []
        mse_train = []
        shd_trian = []

        encoder.train()
        decoder.train()
        scheduler.step()

        # update optimizer
        optimizer, lr = update_optimizer(optimizer, CONFIG.lr, c_A)

        enc_times, dec_times, loss_times, back_times = [], [], [], []

        for i in range(1):
            data = train_data[i * data_sample_size : (i + 1) * data_sample_size]
            data = torch.tensor(data.to_numpy().reshape(data_sample_size, data_variable_size, 1))
            if CONFIG.cuda:
                data = data.cuda()
            data = Variable(data).double()

            optimizer.zero_grad()

            t0 = time.time()
            logits= encoder(
                data
            )  # logits is of size: [num_sims, z_dims]
            edges = logits
            enc_times.append(time.time() - t0)

            t0 = time.time()
            # print(origin_A)
            output = decoder(
                edges #data, edges, data_variable_size * CONFIG.x_dims, origin_A, adj_A_tilt_encoder, Wa
            )
            dec_times.append(time.time() - t0)
            if torch.sum(output != output):
                print("nan error\n")

            target = data
            preds = output
            variance = 0.0
            
            t0 = time.time()
            # reconstruction accuracy loss
            loss_nll = nll_gaussian(preds, target, variance)

            # KL loss
            loss_kl = kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll
            # add A loss
            #one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
            #sparse_loss = CONFIG.tau_A * torch.sum(torch.abs(one_adj_A))

            # other loss term
            if CONFIG.use_A_connect_loss:
                connect_gap = A_connect_loss(one_adj_A, CONFIG.graph_threshold, z_gap)
                loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

            if CONFIG.use_A_positiver_loss:
                positive_gap = A_positive_loss(one_adj_A, z_positive)
                loss += 0.1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

            # compute h(A)
            #h_A = _h_A(origin_A, data_variable_size)
            #loss += (
            #    lambda_A * h_A
            #    + 0.5 * c_A * h_A * h_A
            #    + 100.0 * torch.trace(origin_A * origin_A)
            #    #+ sparse_loss
            #)  # +  0.01 * torch.sum(variance * variance)
            loss_times.append(time.time() - t0)


            t0 = time.time()
            # print(loss)
            loss.backward()
            loss = optimizer.step()

            #myA.data = stau(myA.data, CONFIG.tau_A * lr)
            
            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())
            back_times.append(time.time() - t0)
            # store into global stats
    
        if timing_stats is not None:
            timing_stats["enc"].append(np.mean(enc_times))
            timing_stats["dec"].append(np.mean(dec_times))
            timing_stats["loss"].append(np.mean(loss_times))
            timing_stats["back"].append(np.mean(back_times))
        
        return (
            np.mean(np.mean(kl_train) + np.mean(nll_train)),
            np.mean(nll_train),
            np.mean(mse_train)
        )

    #class Config: pass
    #config = Config()
    #config.win_size = 47  # Window size
    #config.DSR = 1  # Downsampling rate
    #config.cutfreq = 0  # Cut frequency for FITS, set to 0 for automatic calculation
    #if config.cutfreq == 0:
    #    config.cutfreq = int((config.win_size / config.DSR)/2)
    #assert (config.win_size / config.DSR)/2 >= config.cutfreq, 'cutfreq should be smaller than half of the window size after downsampling'
    #
    #config.seq_len = config.win_size//config.DSR
    #config.pred_len = 0#config.win_size-config.win_size//config.DSR
    #config.individual	= False  
    #config.enc_in = 64
    class Config: pass
    config = Config()
    config.seq_len = 47       # match input sequence length
    config.pred_len = 0       # reconstruction only
    config.individual = False
    config.enc_in = 1         # number of input channels (since input.shape[-1] = 1)

    fits = FITSModel(config).to(device)

    # ===================================
    # training: simplified AE (reconstruction only)
    # ===================================
    def train(epoch, optimizer, batch_size=64):
        #encoder.train()
        #decoder.train()

        # prepare dataset
        tensor_data = torch.tensor(train_data.to_numpy(), dtype=torch.float32).to(device)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        total_mse = 0.0
        num_samples = 0

        for batch in loader:
            x = batch[0]

            # forward
            data = x.reshape(x.shape[0], x.shape[1], 1)



            #x = Variable(data).double()
            #z = encoder(x)
            #x_recon = decoder(z)
            # use FITS encoder/decoder
            x_recon = fits(data)    
            # loss = reconstruction error
            loss = F.mse_loss(x_recon.squeeze(-1), x)
            # reconstruction accuracy loss
            #loss_nll = nll_gaussian(x_recon.squeeze(-1), x, 0)

            # KL loss
            #loss_kl = kl_gaussian_sem(z)

            # ELBO loss:
            #loss += loss_nll
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)
            total_mse += loss.item() * len(x)
            num_samples += len(x)

        avg_loss = total_loss / num_samples
        avg_mse = total_mse / num_samples

        print(f"Epoch {epoch:03d} | Loss={avg_loss:.6f}")

        return avg_loss, avg_mse

    # ===================================
    # main
    # ===================================

    # gamma = 0.5
    gamma = 0.25
    eta = 10

    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # optimizer step on hyparameters
    c_A = CONFIG.c_A
    lambda_A = CONFIG.lambda_A
    h_A_new = torch.tensor(1.0)
    h_tol = CONFIG.h_tol
    k_max_iter = int(CONFIG.k_max_iter)
    h_A_old = np.inf

    E_loss = []
    N_loss = []
    M_loss = []
    start_time = time.time()
    # name of experiment for TensorBoard logging
    
    exp_name = "causalrca_experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"./runs/{exp_name}")
    try:
        for step_k in range(k_max_iter):
            # print(step_k)
            while c_A < 1e20:
                for epoch in range(CONFIG.epochs):
                    # print(epoch)
                    # only log first epoch
                    avg_loss, avg_mse = train(
                        epoch, optimizer
                    )
                    # ----------------- TensorBoard Logging -----------------
                    if writer is not None:
                        writer.add_scalar("loss/ELBO", avg_loss, step_k * CONFIG.epochs + epoch)
                        writer.add_scalar("loss/MSE", avg_mse, step_k * CONFIG.epochs + epoch)
                        writer.add_scalar("h_A", h_A_new.item(), step_k * CONFIG.epochs + epoch)
                        writer.add_scalar("c_A", c_A, step_k * CONFIG.epochs + epoch)
                        writer.add_scalar("lambda_A", lambda_A, step_k * CONFIG.epochs + epoch)
                    # -------------------------------------------------------
                        

                    # print(f"{ELBO_loss=} {NLL_loss=} {MSE_loss=}")
                    E_loss.append(avg_loss)
                    N_loss.append(avg_mse)
                    M_loss.append(avg_mse)
                    if avg_loss < best_ELBO_loss:
                        best_ELBO_loss = avg_loss
                        best_epoch = epoch

                    if avg_mse < best_NLL_loss:
                        best_NLL_loss = avg_mse
                        best_epoch = epoch

                    if avg_mse < best_MSE_loss:
                        best_MSE_loss = avg_mse
                        best_epoch = epoch

                # print("Optimization Finished!")
                # print("Best Epoch: {:04d}".format(best_epoch))
                if avg_loss > 2 * best_ELBO_loss:
                    break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()

            if h_A_new.item() <= h_tol:
                break
            # after finishing this step_k
        print(f"[step_k={step_k}] "
            f"avg encoder={np.mean(timing_stats['enc']):.4f}s, "
            f"avg decoder={np.mean(timing_stats['dec']):.4f}s, "
            f"avg loss={np.mean(timing_stats['loss']):.4f}s, "
            f"avg backward={np.mean(timing_stats['back']):.4f}s"
        )
        

    except KeyboardInterrupt:
        print("Done!")

    writer.close()
    # just build on my code to apply POT for ranks 
    # ===================================
    # After training → reconstruction & POT scoring
    # ===================================
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # full dataset
        # safe reshape & dtype & device
        data_np = train_data.to_numpy()
        assert data_np.ndim == 2
        data_tensor = torch.tensor(
            data_np.reshape(data_sample_size, data_variable_size, 1),
            dtype=torch.double if next(encoder.parameters()).dtype == torch.double else torch.float,
            device=device,
        )
        # forward
        try:
            z = encoder(data_tensor)
            preds = decoder(z)
        except Exception as e:
            # If encoder/decoder signatures differ, try alternate call signatures
            # fallback: attempt encoder(x) -> z ; decoder(z) -> preds OR encoder returns preds directly
            try:
                preds = encoder(data_tensor)  # maybe encoder is an AE
                print("Warning: used encoder output as preds fallback")
            except Exception as e2:
                print("ERROR: encoder/decoder forward failed:", e, e2)
                raise

        preds_np = preds.detach().cpu().numpy().reshape(data_sample_size, data_variable_size)
        target_np = data_tensor.detach().cpu().numpy().reshape(data_sample_size, data_variable_size)

        # sanity checks & diagnostics
        if not (preds_np.shape == target_np.shape):
            print("SHAPE MISMATCH: preds", preds_np.shape, "target", target_np.shape)
        if np.isnan(preds_np).any() or np.isnan(target_np).any():
            print("NAN detected in preds or targets:", np.isnan(preds_np).sum(), np.isnan(target_np).sum())

        # per-variable reconstruction error (MSE across time)
        recon_error = np.mean((preds_np - target_np) ** 2, axis=0)  # length = num_vars

        # diagnostic prints
        print("recon_error stats -> mean: {:.6e}, std: {:.6e}, min: {:.6e}, max: {:.6e}".format(
            recon_error.mean(), recon_error.std(), recon_error.min(), recon_error.max()
        ))
        # optionally print top few variables by error
        top_idx = np.argsort(recon_error)[-10:][::-1]
        print("top recon_error indices:", top_idx, "values:", recon_error[top_idx])

        # === apply POT thresholding ===
        # try a safer set of POT params; if POT fails or returns degenerate result, fallback to recon ranking
        pot_risk = getattr(CONFIG, "pot_risk", 1e-2)
        pot_init = getattr(CONFIG, "pot_init_level", 0.98)
        pot_num_candidates = getattr(CONFIG, "pot_num_candidates", 10)
        pot_epsilon = getattr(CONFIG, "pot_epsilon", 1e-8)

        # relax init_level a bit to be less strict if recon_error variance is low
        if recon_error.std() < 1e-8:
            print("Low recon_error variance detected; lowering init_level and increasing risk for POT.")
            pot_init = min(0.95, pot_init)
            pot_risk = max(1e-2, pot_risk * 10)

        try:
            scores, pot_thresh, pot_info = compute_pot_scores(
                recon_error,
                risk=pot_risk,
                init_level=pot_init,
                num_candidates=pot_num_candidates,
                epsilon=pot_epsilon,
            )
        except Exception as e:
            print("compute_pot_scores crashed:", e)
            scores, pot_thresh, pot_info = None, None, None

        # If POT failed or gave degenerate threshold, fallback to direct ranking by recon_error
        if scores is None or (pot_thresh is None) or (np.isnan(pot_thresh)) or (len(scores) != len(recon_error)) or np.allclose(recon_error, recon_error[0]):
            print("POT unsuccessful or degenerate — falling back to reconstruction-error ranking.")
            # simple score = recon_error normalized
            if recon_error.max() - recon_error.min() > 0:
                scores = ((recon_error - recon_error.min()) / (recon_error.max() - recon_error.min())).tolist()
            else:
                scores = recon_error.tolist()
            # set threshold to top-k or small value
            pot_thresh = float(np.percentile(recon_error, 90)) if recon_error.size > 0 else 0.0

        # build ranking of variables
        node_names = train_data.columns.to_list()
        ranks = list(zip(node_names, scores))
        ranks.sort(key=lambda x: x[1], reverse=True)
        ranks = [x[0] for x in ranks]

    # final return dict
    return {
        "scores": np.array(scores).tolist(),
        "threshold": float(pot_thresh),
        "ranks": ranks,
        "node_names": node_names,
    }


if __name__ == "__main__":
    data = pd.read_csv("/home/luan/ws/cfm/tmp_data/cartservice_mem/1/data.csv")

    n = 30

    # read inject_time
    with open("/home/luan/ws/cfm/tmp_data/cartservice_mem/1/inject_time.txt", "r") as f:
        inject_time = f.read()
    inject_time = int(inject_time)
    normal_df = data[data["time"] <= inject_time].tail(n)
    anomalous_df = data[data["time"] > inject_time].head(n)
    data = pd.concat([normal_df, anomalous_df], ignore_index=True)

    output = causalrca(data, inject_time=None, dataset="ob")
    print(output)