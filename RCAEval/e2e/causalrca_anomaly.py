from csv import writer
import math
import time
import warnings
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch.optim as optim
from sknetwork.ranking import PageRank
from torch.optim import lr_scheduler
from RCAEval.io.time_series import preprocess, drop_constant
from utils import compute_kl_divergence
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
        h = F.relu(self.fc1(z.double()  ))
        h = self.dropout(h)
        x_recon = self.fc2(h)
        return x_recon



class AttentionCoeffGNN_multihead_fixed(nn.Module):
    def __init__(self, num_vars, rank, hidden_dim=128, heads=4, extra_layers=1):
        """
        Multi-head attention coefficient generator (fixed version).
        - Avoids mean pooling to preserve mid-ranked signals.
        - Adds residual connection to maintain weaker correlations.
        """
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.heads = heads
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.head_dim = hidden_dim // heads

        # Q, K, V projections
        self.q = nn.Linear(1, hidden_dim)
        self.k = nn.Linear(1, hidden_dim)
        self.v = nn.Linear(1, hidden_dim)

        # LayerNorm after attention
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Residual projection from input
        self.residual = nn.Linear(num_vars, hidden_dim)

        # Build MLP as nn.Sequential (simpler than ModuleList loop)
        mlp_layers = [nn.Linear(num_vars * hidden_dim, hidden_dim), nn.ReLU()]
        for _ in range(extra_layers):
            mlp_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        mlp_layers.append(nn.Linear(hidden_dim, 2 * num_vars * rank))
        self.mlp = nn.Sequential(*mlp_layers)

        # Optional scaling parameter
        self.global_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, attn_dropout=0.1):
        """
        x: (B, p)
        returns: coeffs_k: (B, p, p)
        """
        B, p = x.shape
        x_unsq = x.unsqueeze(-1)  # (B, p, 1)

        # Project Q, K, V
        Q = self.q(x_unsq)
        K = self.k(x_unsq)
        V = self.v(x_unsq)

        # Split into heads
        Q = Q.view(B, p, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, p, head_dim)
        K = K.view(B, p, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, p, self.heads, self.head_dim).transpose(1, 2)

        # Head-wise LayerNorm
        Q = F.layer_norm(Q, (self.head_dim,))
        K = F.layer_norm(K, (self.head_dim,))
        V = F.layer_norm(V, (self.head_dim,))

        # Attention
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.num_vars ** 0.5)
        tau = 0.3  # temperature for sharper attention
        attn_weights = F.softmax(attn_logits / tau, dim=-1)  # try tau ∈ {0.3, 0.5, 0.7, 1.0}

        # Attention dropout
        attn_weights = F.dropout(attn_weights, p=attn_dropout, training=self.training)

        h = torch.matmul(attn_weights, V)  # (B, heads, p, head_dim)

        # Merge heads
        h = h.transpose(1, 2).contiguous().view(B, p, self.heads * self.head_dim)  # (B, p, hidden_dim)

        # Residual connection with scaling
        res = self.residual(x).unsqueeze(1)  # (B, 1, hidden_dim)
        alpha = 0.9  # scale residual to stabilize
        h = h + alpha * res  # broadcast over p dimension

        # LayerNorm over hidden_dim after residual
        h = self.norm1(h)

        # Flatten across variables before MLP
        h_flat = h.view(B, -1)
        h_mlp = self.mlp(h_flat)

        # Split into U and V
        U_flat, V_flat = torch.split(h_mlp, self.num_vars * self.rank, dim=1)
        U = U_flat.view(B, self.num_vars, self.rank)
        V = V_flat.view(B, self.num_vars, self.rank)

        # Reconstruct coefficient matrix
        coeffs_k = torch.bmm(U, V.transpose(1, 2))
        attn_mean = attn_weights.mean(dim=1)
        return coeffs_k,attn_mean

class LinearCoeffGNN(nn.Module):
    def __init__(self, num_vars, rank, hidden_dim=128, heads=4, feat_dim=64, mem_size=16):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.feat_dim = feat_dim
        self.mem_size = mem_size

        self.q = nn.Linear(1, hidden_dim)
        self.k = nn.Linear(1, hidden_dim)
        self.v = nn.Linear(1, hidden_dim)

        # Memory projection (for compressing global K/V)
        self.mem_proj = nn.Linear(self.head_dim, self.mem_size, bias=False)

        # Per-variable U and V producers
        self.project_to_U = nn.Linear(hidden_dim, rank)
        self.project_to_V = nn.Linear(hidden_dim, rank)

    def forward(self, x):
        B, p = x.shape
        x_unsq = x.unsqueeze(-1)
        Q = self.q(x_unsq)
        K = self.k(x_unsq)
        V = self.v(x_unsq)
        # reshape and split heads
        Q = Q.view(B, p, self.heads, self.head_dim).transpose(1,2)
        K = K.view(B, p, self.heads, self.head_dim).transpose(1,2)
        V = V.view(B, p, self.heads, self.head_dim).transpose(1,2)

        # apply feature map φ: e.g. elu + 1 (positive)
        def phi(u):
            return F.elu(u) + 1  # e.g. positive map

        Qf = phi(Q)  # (B, H, p, head_dim)
        Kf = phi(K)  # (B, H, p, head_dim)

        # compress Kf & V into memory: for each head
        # (B, H, mem_size, head_dim) = (B,H,mem_size,p) × (B,H,p,head_dim)
        # Use mem_proj to get weights projecting p → mem_size
        # first, for Kf: project features to memory weights
        # mem_proj maps head_dim → mem_size: acts per token
        W_mem = self.mem_proj(K)  # (B, H, p, mem_size)
        W_mem = F.softmax(W_mem, dim=2)  # over p dimension: which tokens go to each memory slot

        # Now build memory representations
        # mem_KV: (B, H, mem_size, head_dim)
        mem_KV = torch.einsum('bhpm, bhpd -> bhmd', W_mem, V)

        # Attend: Qf @ (mem_KV)^T
        attn_out = torch.matmul(Qf, mem_KV.transpose(-2, -1))  # (B, H, p, mem_size)
        # Then map back: attn_out @ mem_KV → (B,H,p,head_dim)
        h = torch.matmul(attn_out, mem_KV)  # (B, H, p, head_dim)

        # merge heads
        h = h.transpose(1,2).contiguous().view(B, p, self.heads * self.head_dim)

        # produce U, V
        U = self.project_to_U(h)
        V = self.project_to_V(h)

        coeffs_k = torch.bmm(U, V.transpose(1,2))
        return coeffs_k,None

import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentAttentionGNN_Attn(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.device = device
        self.pe_scale = pe_scale

        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)
        self.coeff_proj = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape
        device = inputs.device
        preds_list, coeffs_list = [], []

        pos_embeddings = self.pos_enc[:, :O, :]

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]
            B_chunk = x_chunk.size(0)

            # --- Per-lag GNN features ---
            features_seq = []
            for k in range(O):
                feat_k, _ = self.base_net(x_chunk[:, k, :])  # (B_chunk, P, P)
                features_seq.append(feat_k.reshape(B_chunk, -1))
            features_seq = torch.stack(features_seq, dim=1)  # (B_chunk, O, P*P)

            attn_input = self.in_proj(features_seq) + pos_embeddings  # (B_chunk, O, hidden_dim)

            # --- Multihead scaled-dot attention ---
            head_dim = self.hidden_dim // self.num_heads
            Q = attn_input.view(B_chunk, O, self.num_heads, head_dim).transpose(1, 2)
            K = Q
            V = Q
            dropout_p = 0.1 if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dropout_p, is_causal=False)
            attn_out = attn_out.transpose(1, 2).reshape(B_chunk, O, self.hidden_dim)

            # --- Back to coefficients ---
            coeffs_seq = self.coeff_proj(attn_out).view(B_chunk, O, P, P)

            # --- Predict ---
            preds_chunk = sum(
                torch.matmul(coeffs_seq[:, k, :, :], x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)
                for k in range(O)
            )

            preds_list.append(preds_chunk)
            coeffs_list.append(coeffs_seq)

        preds = torch.cat(preds_list, dim=0)
        coeffs = torch.cat(coeffs_list, dim=0)
        return preds, coeffs


class Decoder_attnn(nn.Module):
    def __init__(self, num_vars, rank, hidden_layer_size=128, device="cpu"):
        super().__init__()
        hidden_dim_small = min(hidden_layer_size, 64)  # smaller hidden dim to reduce parameters
        self.rank = 1                 # low-rank for coefficient matrices

        self.decoding_input_proj = nn.Linear(num_vars, hidden_dim_small).to(device)

        self.decoding_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim_small, num_heads=2, batch_first=True
        ).to(device)

        self.decoding_norm = nn.LayerNorm(hidden_dim_small).to(device)

        self.temporal_attn_decoder = nn.MultiheadAttention(
            embed_dim=hidden_dim_small, num_heads=1, batch_first=True
        ).to(device)

        self.decoding_output_proj = nn.Linear(hidden_dim_small, num_vars).to(device)

        self.decoding_coeff_proj = nn.Linear(hidden_dim_small, 2 * num_vars * rank).to(device)  
        # produces U and V for low-rank coeffs

        self.coeff_proj_decoder = nn.Linear(hidden_dim_small, 2 * num_vars * rank).to(device)   

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 10000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list = []
        coeffs_list = []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)
            B_chunk = x_chunk.size(0)

            # --- Step 1: project inputs ---
            proj_input = self.decoding_input_proj(x_chunk)  # (B_chunk, O, hidden_dim_small)

            # --- Step 2: self-attention ---
            attn_out, _ = self.decoding_attn(proj_input, proj_input, proj_input)  # (B_chunk, O, hidden_dim_small)
            attn_out = self.decoding_norm(attn_out + proj_input)  # residual + norm

            # --- Step 3: temporal attention ---
            temp_attn_out, _ = self.temporal_attn_decoder(attn_out, attn_out, attn_out)  # (B_chunk, O, hidden_dim_small)

            # --- Step 4: output projection ---
            preds_chunk = self.decoding_output_proj(temp_attn_out)  # (B_chunk, O, P)

            # --- Step 5: coefficient generation ---
            coeffs_mlp = self.decoding_coeff_proj(temp_attn_out)    # (B_chunk, O, 2*P*rank)
            U_flat, V_flat = torch.split(coeffs_mlp, P * self.rank, dim=2)
            U = U_flat.view(B_chunk, O, P, self.rank)
            V = V_flat.view(B_chunk, O, P, self.rank)
            coeffs_seq = torch.matmul(U, V.transpose(2, 3))         # (B_chunk, O, P, P)

            preds_list.append(preds_chunk)
            coeffs_list.append(coeffs_seq)

        preds = torch.cat(preds_list, dim=0)                    # (B, O, P)
        coeffs = torch.cat(coeffs_list, dim=0)                  # (B, O, P, P)

        return preds, coeffs



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
    epochs = 50

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
def compute_pot_scores(errors, risk=1e-2, init_level=0.98, num_candidates=100, epsilon=1e-8):
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
                   risk=risk,
                   init_level=init_level,
                   num_candidates=num_candidates,
                   epsilon=epsilon)
    except Exception as e:
        # Fallback: simple quantile threshold
        z = np.quantile(errors, 1 - risk)
        t = np.where(errors > z)[0]

    # Normalize scores
    scores = np.maximum(0, errors - z)
    if scores.max() > 0:
        scores = scores / scores.max()

    return scores, z, t


def _sparsity_loss(coeffs, alpha):
    norm2 = torch.mean(torch.norm(coeffs, dim=1, p=2))
    norm1 = torch.mean(torch.norm(coeffs, dim=1, p=1))
    return (1 - alpha) * norm2 + alpha * norm1


# =========================
# Main causal RCA function
# =========================
def causalrca_anomaly(data, inject_time=None, dataset=None, with_bg=False, with_baro=False, **kwargs):

    print(f"with_baro={with_baro}")


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

    #encoder = MLPEncoder(
    #    data_variable_size * CONFIG.x_dims,
    #    CONFIG.x_dims,
    #    CONFIG.encoder_hidden,
    #    int(CONFIG.z_dims),
    #    adj_A,
    #    batch_size=CONFIG.batch_size,
    #    do_prob=CONFIG.encoder_dropout,
    #    factor=CONFIG.factor,
    #).double()
    encoder = RecurrentAttentionGNN_Attn(
        num_vars=data_variable_size * CONFIG.x_dims,
        rank=51,
        order=1,
        hidden_dim=128,
        num_heads=2,
        device="cuda" if CONFIG.cuda else "cpu",
        attention_heads=4,
        attention_dim=64,
        pe_scale=0.01
    ).double()

    decoder = MLPDecoder(
        data_variable_size * CONFIG.x_dims,
        data_variable_size * CONFIG.x_dims,
        data_variable_size * CONFIG.x_dims,
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


    # ===================================
    # training: simplified AE (reconstruction only)
    # ===================================
    def train(epoch, optimizer, batch_size=64):
        encoder.train()
        decoder.train()

        # Create input and next-step pairs
        x_data = train_data[:-1]      # all except last
        next_data = train_data[1:]    # all except first

        tensor_x = torch.tensor(x_data.to_numpy(), dtype=torch.float32).to(device)
        tensor_next = torch.tensor(next_data.to_numpy(), dtype=torch.float32).to(device)

        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_next)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        mse_loss = nn.MSELoss()
        total_loss = 0.0
        total_mse = 0.0
        num_samples = 0
        for batch in loader:
            # forward
            x, next = batch
            x = Variable(x).double()
            next = Variable(next).double()

            # x make it 3d with time lag 1
            # batch, time lag = 1, num_vars
            x = x.view(x.size(0), 1, x.size(1))
            next = next.view(next.size(0), 1, next.size(1))
            preds, coeff = encoder(x)
            z = preds - next.squeeze(1)
            x_recon = decoder(z)

            # reconstruction accuracy loss
            #loss_nll = nll_gaussian(x_recon, x.squeeze(1), 0.0)
            loss_mse = mse_loss(x_recon.squeeze(1), next.squeeze(1))
            loss_encoder_coeffs = _sparsity_loss(coeff, 0.5)# encoder_alpha = 0.5
            kl_div = compute_kl_divergence(z, device)
            # KL loss
            loss_kl = kl_gaussian_sem(z)

            # ELBO loss:
            loss = loss_mse + loss_kl

            #loss = loss_kl + loss_mse #+ 0.1 * kl_div
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
            #while c_A < 1e20:
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
        data_np = train_data.to_numpy()
        data_tensor = torch.tensor(
            train_data.to_numpy().reshape(data_sample_size, 1, data_variable_size),
            dtype=torch.double,
            device=device,
        )
        # === build the next-step tensor (shifted version) ===
        next_np = np.roll(data_np, shift=-1, axis=0)  # shift forward in time
        next_np[-1, :] = next_np[-2, :]               # copy last row to avoid wrapping
        next_tensor = torch.tensor(
            next_np.reshape(data_sample_size, 1, data_variable_size),
            dtype=torch.double,
            device=device,
        )

        batch_size = 64  # or smaller depending on GPU

        preds_list = []
        coeff_list = []

        for start in range(0, data_tensor.size(0), batch_size):
            end = start + batch_size
            x_batch = data_tensor[start:end]

            with torch.no_grad():
                preds_batch, coeff_batch = encoder(x_batch)

            preds_list.append(preds_batch.cpu())  # move to CPU to save GPU memory
            coeff_list.append(coeff_batch.cpu())  # optional: store on CPU if needed

        preds = torch.cat(preds_list, dim=0)
        coeff = torch.cat(coeff_list, dim=0)
        # now move preds back to device
        preds = preds.to(device)
        coeff = coeff.to(device)
        z = preds - next_tensor.squeeze(1)

        preds = decoder(z)
        preds_np = preds.detach().cpu().numpy().reshape(data_sample_size, data_variable_size)
        target_np = data_tensor.detach().cpu().numpy().reshape(data_sample_size, data_variable_size)

        # === Latent deviation normalization (encoder–decoder) ===
        us_np = z.detach().cpu().numpy().reshape(data_sample_size, data_variable_size)
        us_mean = np.mean(us_np, axis=0)
        us_std = np.std(us_np, axis=0) + 1e-8
        us_z_score = (-(us_np - us_mean) / us_std)  # shape: (T, num_vars)

        node_names = train_data.columns.to_list()
        data_variable_size = len(node_names)

        # === POT threshold per variable (latent / encoder–decoder) ===
        scores = []
        for i in range(data_variable_size):
            pot_val, _, _ = compute_pot_scores(
                us_z_score[:, i],
                risk=getattr(CONFIG, "pot_risk", 1e-2),
                init_level=getattr(CONFIG, "pot_init_level", 0.98),
                num_candidates=getattr(CONFIG, "pot_num_candidates", 10),
                epsilon=getattr(CONFIG, "pot_epsilon", 1e-8),
            )
            scores.append(pot_val)
        scores = np.array(scores)
        ed_scores = np.array([val.mean() for val in scores])  # mean anomaly fraction per var

        if with_baro:
            # === BARO-style statistical deviation per variable ===
            # Split into "normal" and "anomalous" parts — you can use your inject_time or anomaly split logic\
            split_idx = len(train_data) // 2  # fallback simple split
            normal_df = train_data.iloc[:split_idx]
            anomal_df = train_data.iloc[split_idx:]

            baro_scores = []
            for col in node_names:
                a = normal_df[col].to_numpy()
                b = anomal_df[col].to_numpy()
                scaler = RobustScaler().fit(a.reshape(-1, 1))
                zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
                baro_scores.append(np.max(zscores))  # BARO = max deviation
            baro_scores = np.array(baro_scores)

            # === Combine both (hybrid fusion) ===
            alpha = 0.6  # weight for encoder–decoder signal
            hybrid_scores = alpha * ed_scores + (1 - alpha) * baro_scores
        else:
            baro_scores = np.zeros_like(ed_scores)
            hybrid_scores = ed_scores
        # === Rank variables ===
        ranks = list(zip(node_names, hybrid_scores))
        ranks.sort(key=lambda x: x[1], reverse=True)
        ranks = [x[0] for x in ranks]

    # === Final return dict ===
    return {
        "scores": hybrid_scores.tolist(),
        "ranks": ranks,
        "node_names": node_names,
        "ed_scores": ed_scores.tolist(),
        "baro_scores": baro_scores.tolist(),
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