import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import numpy as np
from numpy.polynomial.legendre import legfit, legval

from layers.TexFilter import TexFilter

class SimpleCoeffGNN(nn.Module):
    def __init__(self, num_vars: int, rank: int, hidden_dim: int = 128, residual_scale: float = 0.1):
        """
        Simple GNN coefficient generator for one lag with low-rank factors.
        - num_vars: number of variables (p)
        - rank: low-rank factor size
        - hidden_dim: hidden layer size of MLP
        - residual_scale: scaling factor for residual identity
        """
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.residual_scale = residual_scale

        # Learnable adjacency, small initialization
        self.adj = nn.Parameter(torch.randn(num_vars, num_vars) * 0.1)

        # 2-layer MLP for low-rank factors
        self.fc1 = nn.Linear(num_vars, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * num_vars * rank)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc2.bias)

    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    def forward(self, x):
        """
        x: (B, p) input at one lag
        returns: coeffs_k (B, p, p)
        """
        # Row-normalized adjacency to prevent exploding messages
        adj_norm = F.softmax(self.adj, dim=1)
        h = torch.matmul(adj_norm, x.unsqueeze(2)).squeeze(2)  # (B, p)

        # Optional clamping to avoid huge values
        h = torch.clamp(h, -10.0, 10.0)

        # 2-layer MLP
        h = F.relu(self.fc1(h))
        out = self.fc2(h)

        # Split into U and V for low-rank reconstruction
        U_flat, V_flat = torch.split(out, self.num_vars * self.rank, dim=1)
        U = U_flat.view(-1, self.num_vars, self.rank)
        V = V_flat.view(-1, self.num_vars, self.rank)
        coeffs_k = torch.bmm(U, V.transpose(1, 2))  # (B, p, p)


        return coeffs_k
    

class AttentionCoeffGNN(nn.Module):
    def __init__(self, num_vars: int, rank: int, hidden_dim: int = 128):
        """
        Attention-based GNN coefficient generator for one lag.
        - num_vars: number of variables (p)
        - rank: low-rank factor size
        - hidden_dim: hidden layer size for attention MLP
        """
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank

        # Linear layers to compute queries, keys, values
        self.q = nn.Linear(num_vars, hidden_dim)
        self.k = nn.Linear(num_vars, hidden_dim)
        self.v = nn.Linear(num_vars, hidden_dim)

        # MLP to project aggregated features to U and V
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * num_vars * rank)

        # Optional scaling parameter
        self.global_scale = nn.Parameter(torch.tensor(0.1))

    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)
            
    def forward(self, x):
        """
        x: (B, p) input at one lag
        returns: coeffs_k (B, p, p)
        """
        B, p = x.shape

        # Compute Q, K, V
        Q = self.q(x)       # (B, num_vars)
        K = self.k(x)       # (B, num_vars)
        V = self.v(x)       # (B, num_vars)
        attn_logits = torch.bmm(Q.unsqueeze(2), K.unsqueeze(1)) / (self.num_vars ** 0.5)  # (B, num_vars, num_vars)
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Aggregate values
        h = torch.bmm(attn_weights, V.unsqueeze(2)).squeeze(2)  # (B, hidden_dim)

        # 2-layer MLP to predict low-rank factors
        h = F.relu(self.fc1(h))
        out = self.fc2(h)

        # Split into U and V and reconstruct coefficient matrix
        U_flat, V_flat = torch.split(out, self.num_vars * self.rank, dim=1)
        U = U_flat.view(-1, self.num_vars, self.rank)
        V = V_flat.view(-1, self.num_vars, self.rank)
        coeffs_k = torch.bmm(U, V.transpose(1, 2))

        # Optional residual identity + scaling
        #coeffs_k = coeffs_k * self.global_scale + torch.eye(self.num_vars, device=x.device)

        return coeffs_k



class AttentionCoeffGNN_multihead(nn.Module):
    def __init__(self, num_vars, rank, hidden_dim=16, heads=2, extra_layers=1):
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

        # Build MLP as one ModuleList
        mlp_layers = []
        mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        mlp_layers.append(nn.ReLU())
        for _ in range(extra_layers):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(hidden_dim, 2 * num_vars * rank))
        self.mlp = nn.ModuleList(mlp_layers)

        # Optional scaling parameter
        self.global_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        x: (B, p)
        returns: coeffs_k: (B, p, p)
        """
        B, p = x.shape

        x_unsq = x.unsqueeze(-1)  # (B, p, 1)

        Q = self.q(x_unsq)  # (B, p, hidden_dim)
        K = self.k(x_unsq)  # (B, p, hidden_dim)
        V = self.v(x_unsq)  # (B, p, hidden_dim)

        # Split into heads
        Q = Q.view(B, p, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, p, head_dim)
        K = K.view(B, p, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, p, head_dim)
        V = V.view(B, p, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, p, head_dim)
        
        # Compute scaled dot-product attention for each head
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.num_vars ** 0.5)  # (B, heads, p, p)
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Aggregate values
        h = torch.matmul(attn_weights, V)  # (B, heads, p, head_dim)

        # Merge heads: (B, p, hidden_dim)
        h = h.transpose(1, 2).contiguous().view(B, p, self.heads * self.head_dim)

        # Mean-pool across p to get global vector: (B, hidden_dim)
        h = h.mean(dim=1)

        # Norm
        h = self.norm1(h)

        # Pass through MLP
        for layer in self.mlp:
            h = layer(h)

        # Split into U and V
        U_flat, V_flat = torch.split(h, self.num_vars * self.rank, dim=1)
        U = U_flat.view(B, self.num_vars, self.rank)
        V = V_flat.view(B, self.num_vars, self.rank)

        # Reconstruct coefficient matrix
        coeffs_k = torch.bmm(U, V.transpose(1, 2))
        return coeffs_k
    
    
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
    
    def _init_weights(self):
        # Initialize Linear layers with Xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

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

class RecurrentAttentionCoeffGNN__(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=64, num_layers=1, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.device = device

        # Shared GNN coefficient extractor per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=rank)

        # RNN across lags
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)  # project flattened coeffs
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Project hidden state to coefficient adjustment
        self.context_proj = nn.Linear(hidden_dim, num_vars * num_vars)
    def _init_weights(self):
        # Initialize Linear layers with Xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, inputs: torch.Tensor):
        """
        inputs: (B, order, num_vars)
        returns:
            preds: (B, num_vars)
            coeffs: (B, order, num_vars, num_vars)
        """
        B, O, P = inputs.shape
        if (O, P) != (self.order, self.num_vars):
            print("WARNING: inputs should be of shape BS x K x p")

        # --- Step 1: compute per-lag coefficients ---
        coeffs_seq = []
        for k in range(O):
            coeff_k = self.base_net(inputs[:, k, :])    # (B, P, P)
            coeffs_seq.append(coeff_k.view(B, -1))      # flatten

        # Sequence: (B, O, P*P)
        coeffs_seq = torch.stack(coeffs_seq, dim=1)

        # --- Step 2: process sequence with RNN ---
        rnn_input = self.in_proj(coeffs_seq)  # (B, O, hidden_dim)
        h_seq, h_final = self.rnn(rnn_input)  # h_seq: (B, O, hidden_dim), h_final: (num_layers, B, hidden_dim)
        h_final = h_final[-1]                 # (B, hidden_dim) last layer final hidden

        # --- Step 3: project hidden state to coefficient adjustment ---
        context_adjust = self.context_proj(h_final).view(B, P, P)  # (B, P, P)

        # --- Step 4: add global context to each lag coefficient ---
        coeffs_rnn = coeffs_seq.view(B, O, P, P) + context_adjust.unsqueeze(1)  # broadcast across lags

        # --- Step 5: compute predictions ---
        preds = torch.zeros((B, P), device=self.device)
        for k in range(O):
            preds += torch.matmul(coeffs_rnn[:, k, :, :], inputs[:, k, :].unsqueeze(-1)).squeeze(-1)

        return preds, coeffs_rnn

class RecurrentAttentionCoeffGNN_chunks(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=32, proj_dim=32, num_layers=1, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.device = device

        # Shared GNN coefficient extractor per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=rank)

        # RNN across lags
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Project hidden state to coefficient adjustment
        self.context_proj = nn.Linear(hidden_dim, num_vars * num_vars)

        # 🔑 Projection for coefficients to smaller space
        self.coeff_proj = nn.Linear(num_vars * num_vars, proj_dim)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 16, return_coeffs: bool = True):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list = []
        coeffs_list = [] if return_coeffs else None

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)
            B_chunk = x_chunk.size(0)

            # --- Step 1: compute per-lag coefficients ---
            coeffs_seq = []
            for k in range(O):
                coeff_k = self.base_net(x_chunk[:, k, :])    # (B_chunk, P, P)
                coeffs_seq.append(coeff_k.view(B_chunk, -1)) # (B_chunk, P*P)

            coeffs_seq = torch.stack(coeffs_seq, dim=1)      # (B_chunk, O, P*P)

            # --- Step 2: process sequence with RNN ---
            rnn_input = self.in_proj(coeffs_seq)             # (B_chunk, O, hidden_dim)
            h_seq, h_final = self.rnn(rnn_input)
            h_final = h_final[-1]                            # (B_chunk, hidden_dim)

            # --- Step 3: project hidden state to coefficient adjustment ---
            context_adjust = self.context_proj(h_final).view(B_chunk, P, P)

            # --- Step 4: add global context ---
            coeffs_rnn = coeffs_seq.view(B_chunk, O, P, P) + context_adjust.unsqueeze(1)

            # --- Step 5: compute predictions ---
            preds_chunk = torch.zeros((B_chunk, P), device=device)
            for k in range(O):
                preds_chunk += torch.matmul(coeffs_rnn[:, k, :, :], x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)

            preds_list.append(preds_chunk)

            if return_coeffs:
                # 🔑 project coeffs to smaller space before storing
                coeffs_proj = self.coeff_proj(coeffs_rnn.view(B_chunk, O, -1))  # (B_chunk, O, proj_dim)
                coeffs_list.append(coeffs_proj)

        preds = torch.cat(preds_list, dim=0)
        coeffs_seq = torch.cat(coeffs_list, dim=0) if return_coeffs else None

        return preds, coeffs_seq
    

class RecurrentAttentionCoeffGNN(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=32, num_layers=1, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.device = device

        # Base network per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=rank)

        # RNN across lags
        self.rnn = nn.GRU(
            input_size=num_vars * num_vars,  # flattened output from base_net
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Project hidden state to prediction
        self.pred_proj = nn.Linear(hidden_dim, num_vars)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list = []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)
            B_chunk = x_chunk.size(0)

            # --- Step 1: extract features per lag ---
            features_seq = []
            for k in range(O):
                feat_k = self.base_net(x_chunk[:, k, :])    # (B_chunk, P, P)
                features_seq.append(feat_k.view(B_chunk, -1))  # flatten to (B_chunk, P*P)

            features_seq = torch.stack(features_seq, dim=1)  # (B_chunk, O, P*P)

            # --- Step 2: process with RNN ---
            _, h_final = self.rnn(features_seq)
            h_final = h_final[-1]  # (B_chunk, hidden_dim)

            # --- Step 3: predict from hidden state ---
            preds_chunk = self.pred_proj(h_final)  # (B_chunk, P)
            preds_list.append(preds_chunk)

        preds = torch.cat(preds_list, dim=0)  # (B, P)
        return preds, None
    

class RecurrentAttentionGNN_Attn_nocoeff(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=64, num_heads=4, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device

        # Base GNN per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(num_vars=num_vars, rank=rank)

        # Project flattened GNN output to hidden dimension for attention
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)

        # Temporal attention across lags
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Output projection from aggregated hidden state
        self.pred_proj = nn.Linear(hidden_dim, num_vars)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list = []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)
            B_chunk = x_chunk.size(0)

            # --- Step 1: extract features per lag ---
            features_seq = []
            for k in range(O):
                feat_k = self.base_net(x_chunk[:, k, :])     # (B_chunk, P, P)
                features_seq.append(feat_k.view(B_chunk, -1))  # flatten to (B_chunk, P*P)

            features_seq = torch.stack(features_seq, dim=1)   # (B_chunk, O, P*P)

            # --- Step 2: project to hidden_dim for attention ---
            attn_input = self.in_proj(features_seq)          # (B_chunk, O, hidden_dim)

            # --- Step 3: temporal attention ---
            attn_out, _ = self.temporal_attn(attn_input, attn_input, attn_input)  # (B_chunk, O, hidden_dim)

            # --- Step 4: aggregate across lags ---
            # simple mean pooling across lags
            agg_hidden = attn_out.mean(dim=1)               # (B_chunk, hidden_dim)

            # --- Step 5: predict from aggregated hidden state ---
            preds_chunk = self.pred_proj(agg_hidden)        # (B_chunk, P)
            preds_list.append(preds_chunk)

        preds = torch.cat(preds_list, dim=0)               # (B, P)
        return preds, None


class RecurrentAttentionGNN_Attn(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order  # number of lags
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale  # scale factor for positional embedding

        # Base GNN per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # Project flattened GNN output to hidden_dim for attention
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)

        # Temporal attention across lags
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Map attention hidden state back to coefficients
        self.coeff_proj = nn.Linear(hidden_dim, num_vars * num_vars)

        # Learnable positional embedding (small initial scale)
        self.pos_enc = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 10000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list = []
        coeffs_list = []

        # Use positional encoding
        pos_embeddings = self.pos_enc[:, :O, :]  # shape (1, O, hidden_dim)

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)
            B_chunk = x_chunk.size(0)

            # --- Step 1: extract features per lag ---
            features_seq = []
            for k in range(O):
                feat_k, _ = self.base_net(x_chunk[:, k, :])        # (B_chunk, P, P)
                features_seq.append(feat_k.view(B_chunk, -1))   # (B_chunk, P*P)
            features_seq = torch.stack(features_seq, dim=1)      # (B_chunk, O, P*P)

            # --- Step 2: project to hidden_dim for attention + add positional encoding ---
            attn_input = self.in_proj(features_seq) + pos_embeddings  # (B_chunk, O, hidden_dim)

            # --- Step 3: temporal attention ---
            attn_out, _ = self.temporal_attn(attn_input, attn_input, attn_input)  # (B_chunk, O, hidden_dim)

            # --- Step 4: map attention outputs back to coeffs ---
            coeffs_seq = self.coeff_proj(attn_out)               # (B_chunk, O, P*P)
            coeffs_seq = coeffs_seq.view(B_chunk, O, P, P)       # (B_chunk, O, P, P)

            # --- Step 5: prediction using coeffs ---
            preds_chunk = torch.zeros((B_chunk, P), device=device)
            for k in range(O):
                preds_chunk += torch.matmul(coeffs_seq[:, k, :, :], 
                                            x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)

            preds_list.append(preds_chunk)
            coeffs_list.append(coeffs_seq)

        preds = torch.cat(preds_list, dim=0)                    # (B, P)
        coeffs = torch.cat(coeffs_list, dim=0)                  # (B, O, P, P)

        return preds, coeffs, None 


class RecurrentAttentionGNN_Attn_fourier(nn.Module):
    """
    Time domain path: exactly like your current RecurrentAttentionGNN_Attn.
    Freq domain path: rFFT over lags (per variable) -> magnitude -> per-bin GNN -> attention over bins.
    Fusion: gated blend of (time coeffs) and (freq coeffs).
    """
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01, options=None):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale
        self.time_freq_representation = options.get("time_freq_representation", "") # normal, mag_phase, learnable_filter
        self.combine_method = options.get("combine_method", "gated")  # gated, attention

        # Shared per-slice GNN (use your fixed variant; swap if needed)
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # --- Time path ---
        self.in_proj_time = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_time = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_time = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

        # --- Freq path ---
        # rFFT along lags ⇒ F = order//2 + 1 bins
        self.in_proj_freq = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_freq = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_freq = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_freq = nn.Parameter(torch.randn(1, (order // 2) + 1, hidden_dim) * pe_scale)

        if self.combine_method not in ["freq_only", "gated", "sum", "concat"]:
            raise ValueError("combine_method must be 'gated' or 'attention'")
        elif self.combine_method == "gated":
            # --- Gated fusion (global context → α in [0,1]) ---
            # take simple stats from the window as context
            ctx_dim = 2 * num_vars  # mean & std per variable, concatenated
            self.fusion_gate = nn.Sequential(
                nn.Linear(ctx_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        elif self.combine_method == "concat":
            # --- Concatenation fusion (project back to num_vars * num_vars) ---
            self.fusion_proj = nn.Linear(num_vars * num_vars * 2, num_vars * num_vars)
        elif self.combine_method == "sum":
            pass  # no extra layers needed for sum
        
        if self.time_freq_representation == "mag_phase":
            self.in_projector = nn.Linear(num_vars*2, num_vars)
        elif self.time_freq_representation == "mag_phase_learnable_filter":
            self.in_projector = nn.Linear(num_vars*2, num_vars)
            self.texfilter = TexFilter(
                embed_size=num_vars,
                use_gelu=True,             # or use_swish=True for smoother nonlinearity
                use_skip=True,             # ✅ Preserve original signal paths
                use_layernorm=True,        # ✅ Stabilize across frequency bins
                hard_threshold=False,      # ❌ Avoid hard cutting off weak signals
                use_window=False,          # ❌ Avoid muting boundary info
                sparsity_threshold=0.0     # ✅ Retain all weak signal components
            )
        elif self.time_freq_representation == "learnable_filter":
            self.texfilter = TexFilter(
                embed_size=num_vars,
                use_gelu=True,             # or use_swish=True for smoother nonlinearity
                use_skip=True,             # ✅ Preserve original signal paths
                use_layernorm=True,        # ✅ Stabilize across frequency bins
                hard_threshold=False,      # ❌ Avoid hard cutting off weak signals
                use_window=False,          # ❌ Avoid muting boundary info
                sparsity_threshold=0.0     # ✅ Retain all weak signal components
            )

    def _init_weights(self):
        """Initialize weights for linear layers and attention projections."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                # PyTorch MHA already uses xavier for in_proj_weight internally, but can re-init if needed
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
                    
    @torch.no_grad()
    def _context_stats(self, x_win):  # x_win: (B, O, P)
        mean = x_win.mean(dim=1)       # (B, P)
        std = x_win.std(dim=1)         # (B, P)
        return torch.cat([mean, std], dim=-1)  # (B, 2P)

    def _time_path(self, x_chunk):  # x_chunk: (B_chunk, O, P)
        Bc, O, P = x_chunk.shape
        feats = []
        for k in range(O):
            # base_net expects (B, P) and returns (B, P, P)
            coeff_k, _ = self.base_net(x_chunk[:, k, :])            # (Bc, P, P)
            feats.append(coeff_k.view(Bc, -1))                      # (Bc, P*P)
        seq = torch.stack(feats, dim=1)                              # (Bc, O, P*P)
        attn_in = self.in_proj_time(seq) + self.pos_enc_time[:, :O, :]
        attn_out, _ = self.temporal_attn_time(attn_in, attn_in, attn_in)  # (Bc, O, H)
        coeffs_seq = self.coeff_proj_time(attn_out).view(Bc, O, P, P)     # (Bc, O, P, P)
        # prediction using coeffs per lag
        preds = torch.zeros((Bc, P), device=x_chunk.device)
        for k in range(O):
            preds += (coeffs_seq[:, k] @ x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)
        return preds, coeffs_seq

    def _freq_path_normal(self, x_chunk):  # x_chunk: (B_chunk, O, P)
        """
        rFFT over the lag axis (dim=1). We use magnitude per variable & bin: (B, F, P).
        Each frequency bin plays the role of a "lag slice" for base_net.
        """
        Bc, O, P = x_chunk.shape
        X = torch.fft.rfft(x_chunk, dim=1)                     # (Bc, F, P) complex
        Xmag = X.abs()                                         # (Bc, F, P) real magnitudes
        Fbins = Xmag.size(1)

        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(Xmag[:, f, :])          # (Bc, P, P)
            feats.append(coeff_f.view(Bc, -1))                 # (Bc, P*P)
        seq = torch.stack(feats, dim=1)                        # (Bc, F, P*P)

        attn_in = self.in_proj_freq(seq) + self.pos_enc_freq[:, :Fbins, :]
        attn_out, _ = self.temporal_attn_freq(attn_in, attn_in, attn_in)   # (Bc, F, H)
        coeffs_seq = self.coeff_proj_freq(attn_out).view(Bc, Fbins, P, P)  # (Bc, F, P, P)

        # collapse over frequency bins (attention already weighs them; mean works well)
        coeffs_collapsed = coeffs_seq.mean(dim=1)                              # (Bc, P, P)

        # one-step prediction proxy: apply collapsed coeffs to the *latest* lag input
        preds = (coeffs_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)  # (Bc, P)
        return preds, coeffs_seq, coeffs_collapsed
    
    def _freq_path_mag_phase(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape

        # --- 1. rFFT ---
        X = torch.fft.rfft(x_chunk, dim=1)           # (Bc, F, P) complex

        # --- 2. Extract magnitude and phase ---
        X_mag = X.abs()                              # (Bc, F, P)
        X_phase = torch.angle(X)                     # (Bc, F, P)
        X_phase_norm = X_phase / np.pi               # normalize to [-1, 1]

        # --- 3. Concatenate to preserve full signal ---
        X_full = torch.cat([X_mag, X_phase_norm], dim=-1)   # (Bc, F, 2*P)
        Fbins = X_full.size(1)

        # --- 4. GNN per frequency bin ---
        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(self.in_projector(X_full[:, f, :]))
            feats.append(coeff_f.view(Bc, -1))              # flatten (Bc, P*P)
        seq = torch.stack(feats, dim=1)                     # (Bc, F, P*P)

        # --- 5. Temporal attention over frequency bins ---
        attn_in = self.in_proj_freq(seq) + self.pos_enc_freq[:, :Fbins, :]
        attn_out, _ = self.temporal_attn_freq(attn_in, attn_in, attn_in)   # (Bc, F, H)

        # --- 6. Project back to coefficients ---
        coeffs_seq = self.coeff_proj_freq(attn_out).view(Bc, Fbins, P, P)  # (Bc, F, P, P)

        # --- 7. Collapse across frequency bins ---
        coeffs_collapsed = coeffs_seq.mean(dim=1)                           # (Bc, P, P)

        # --- 8. One-step prediction proxy (same as _freq_path_normal) ---
        preds = (coeffs_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)  # (Bc, P)

        return preds, coeffs_seq, coeffs_collapsed#, attn_out, seq, X_full


    def _freq_path_mag_phase_learnable_filter(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape

        # --- 1. rFFT ---
        X = torch.fft.rfft(x_chunk, dim=1)           # (Bc, F, P) complex

        # --- 2. Apply learnable frequency filter (TexFilter) ---
        X = X * self.texfilter(X)                    # element-wise frequency attention

        # --- 3. Extract real and imaginary channels ---
        X_real = X.real                              # (Bc, F, P)
        X_imag = X.imag                              # (Bc, F, P)

        # --- 4. Concatenate to preserve full signal ---
        X_full = torch.cat([X_real, X_imag], dim=-1)  # (Bc, F, 2*P)
        Fbins = X_full.size(1)

        # --- 5. GNN per frequency bin ---
        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(self.in_projector(X_full[:, f, :]))  # (Bc, P, P)
            feats.append(coeff_f.view(Bc, -1))           # flatten to (Bc, P*P)
        seq = torch.stack(feats, dim=1)                  # (Bc, F, P*P)

        # --- 6. Temporal attention over frequency bins ---
        attn_in = self.in_proj_freq(seq) + self.pos_enc_freq[:, :Fbins, :]
        attn_out, _ = self.temporal_attn_freq(attn_in, attn_in, attn_in)  # (Bc, F, H)

        # --- 7. Project back to coefficients ---
        coeffs_seq = self.coeff_proj_freq(attn_out).view(Bc, Fbins, P, P)  # (Bc, F, P, P)

        # --- 8. Collapse across frequency bins ---
        coeffs_collapsed = coeffs_seq.mean(dim=1)                           # (Bc, P, P)

        # --- 9. One-step prediction proxy ---
        preds = (coeffs_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)  # (Bc, P)

        return preds, coeffs_seq, coeffs_collapsed#, attn_out, seq, X_full


    def _freq_path_learnable_filter(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape

        # --- 1. rFFT ---
        X = torch.fft.rfft(x_chunk, dim=1)           # (Bc, F, P) complex

        # --- 2. Apply learnable frequency filter (TexFilter) ---
        X = X * self.texfilter(X)                    # element-wise frequency attention

        # --- 3. Magnitude channel only (filtered) ---
        X_mag = X.abs()                              # (Bc, F, P)
        Fbins = X_mag.size(1)

        # --- 4. GNN per frequency bin ---
        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(X_mag[:, f, :])   # (Bc, P, P)
            feats.append(coeff_f.view(Bc, -1))           # flatten to (Bc, P*P)
        seq = torch.stack(feats, dim=1)                  # (Bc, F, P*P)

        # --- 5. Temporal attention over frequency bins ---
        attn_in = self.in_proj_freq(seq) + self.pos_enc_freq[:, :Fbins, :]
        attn_out, _ = self.temporal_attn_freq(attn_in, attn_in, attn_in)  # (Bc, F, H)

        # --- 6. Project back to coefficients ---
        coeffs_seq = self.coeff_proj_freq(attn_out).view(Bc, Fbins, P, P)  # (Bc, F, P, P)

        # --- 7. Collapse across frequency bins ---
        coeffs_collapsed = coeffs_seq.mean(dim=1)                            # (Bc, P, P)

        # --- 8. One-step prediction proxy ---
        preds = (coeffs_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)  # (Bc, P)

        return preds, coeffs_seq, coeffs_collapsed#, attn_out, seq, X_mag

    


    def _freq_path(self, x_chunk):  # x_chunk: (B_chunk, O, P)
        if self.time_freq_representation == "normal":
            return self._freq_path_normal(x_chunk)
        elif self.time_freq_representation == "mag_phase":
            return self._freq_path_mag_phase(x_chunk)
        elif self.time_freq_representation == "mag_phase_learnable_filter":
            return self._freq_path_mag_phase_learnable_filter(x_chunk)
        elif self.time_freq_representation == "learnable_filter":
            # same as mag_phase_learnable_filter but without phase extraction step
            return self._freq_path_learnable_filter(x_chunk)

    def forward_gated(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_out, coeffs_time_out, coeffs_freq_out = [], [], []
        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)

            # --- time path ---
            preds_t, coeffs_t = self._time_path(x_chunk)             # (Bch,P), (Bch,O,P,P)

            # --- freq path ---
            preds_f, coeffs_f_seq, coeffs_f_collapsed = self._freq_path(x_chunk)  # (Bch,P), (Bch,F,P,P), (Bch,P,P)

            # --- fusion gate ---
            ctx = self._context_stats(x_chunk)                        # (Bch, 2P)
            alpha = self.fusion_gate(ctx)                             # (Bch, 1) in [0,1]

            # fuse predictions (optional; used mainly for training signal)
            preds = alpha * preds_t + (1 - alpha) * preds_f

            # fuse coefficients: time path gives per-lag; freq path is collapsed
            # broadcast alpha to match (Bch, O, 1, 1) and (Bch, 1, 1)
            alpha_time = alpha.view(-1, 1, 1, 1)
            alpha_freq = (1 - alpha).view(-1, 1, 1)

            # add freq coeffs into each lag as a global periodic prior
            alpha_freq = alpha_freq.unsqueeze(-1)  # [131, 1, 1, 1]
            coeffs_fused = alpha_time * coeffs_t + alpha_freq * coeffs_f_collapsed[:, None, :, :]  # (Bch,O,P,P)

            preds_out.append(preds)
            coeffs_time_out.append(coeffs_fused)
            coeffs_freq_out.append(coeffs_f_seq)  # keep raw per-bin seq if you want diagnostics

        preds = torch.cat(preds_out, dim=0)                     # (B, P)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0)    # (B, O, P, P)  (fused)
        coeffs_freq_seq = torch.cat(coeffs_freq_out, dim=0)     # (B, F, P, P)

        return preds, coeffs_time_like, coeffs_freq_seq

    def forward_sum(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape
        preds_out, coeffs_time_out, coeffs_freq_out = [], [], []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)

            # --- time path ---
            preds_t, coeffs_t = self._time_path(x_chunk)  # (Bch, P), (Bch, O, P, P)

            # --- freq path ---
            preds_f, coeffs_f_seq, coeffs_f_collapsed = self._freq_path(x_chunk)  # (Bch,P), (Bch,F,P,P), (Bch,P,P)

            # --- prediction fusion (simple sum) ---
            preds = preds_t + preds_f

            # --- coefficient fusion ---
            # broadcast collapsed freq coeffs across lags
            coeffs_f_broadcast = coeffs_f_collapsed[:, None, :, :].expand(-1, O, -1, -1)
            coeffs_fused = coeffs_t + coeffs_f_broadcast  # (Bch, O, P, P)

            preds_out.append(preds)
            coeffs_time_out.append(coeffs_fused)
            coeffs_freq_out.append(coeffs_f_seq)  # keep raw per-bin seq if needed

        preds = torch.cat(preds_out, dim=0)              # (B, P)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0)  # (B, O, P, P)
        coeffs_freq_seq = torch.cat(coeffs_freq_out, dim=0)   # (B, F, P, P)

        return preds, coeffs_time_like, coeffs_freq_seq

    def forward_concat(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape
        preds_out, coeffs_time_out, coeffs_freq_out = [], [], []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)

            # --- time path ---
            preds_t, coeffs_t = self._time_path(x_chunk)  # (Bch, P), (Bch, O, P, P)

            # --- freq path ---
            preds_f, coeffs_f_seq, coeffs_f_collapsed = self._freq_path(x_chunk)  # (Bch,P), (Bch,F,P,P), (Bch,P,P)

            # --- prediction fusion (optional: sum for training signal) ---
            preds = preds_t + preds_f

            # --- coefficient fusion by concatenation ---
            # flatten coeffs
            coeffs_t_flat = coeffs_t.view(-1, O, P*P)               # (Bch, O, P*P)
            coeffs_f_flat = coeffs_f_collapsed.view(-1, 1, P*P)     # (Bch, 1, P*P)
            coeffs_f_flat = coeffs_f_flat.expand(-1, O, -1)         # (Bch, O, P*P) broadcast across lags

            # concatenate along last dim
            coeffs_concat = torch.cat([coeffs_t_flat, coeffs_f_flat], dim=-1)  # (Bch, O, 2*P*P)
            coeffs_fused = self.fusion_proj(coeffs_concat)                       # (Bch, O, P*P)
            coeffs_fused = coeffs_fused.view(-1, O, P, P)                        # (Bch, O, P, P)

            preds_out.append(preds)
            coeffs_time_out.append(coeffs_fused)
            coeffs_freq_out.append(coeffs_f_seq)  # keep raw per-bin seq if needed

        preds = torch.cat(preds_out, dim=0)              # (B, P)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0)  # (B, O, P, P)
        coeffs_freq_seq = torch.cat(coeffs_freq_out, dim=0)   # (B, F, P, P)

        return preds, coeffs_time_like, coeffs_freq_seq

    def forward_freq_only(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        """
        Frequency-only forward path.
        Uses only the rFFT-based path (normal, mag_phase, learnable_filter, etc.),
        ignoring the time-domain branch.
        """
        B, O, P = inputs.shape
        preds_out, coeffs_time_out, coeffs_freq_out = [], [], []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)

            # --- freq path only ---
            preds_f, coeffs_f_seq, coeffs_f_collapsed = self._freq_path(x_chunk)  # (Bch,P), (Bch,F,P,P), (Bch,P,P)

            # for consistency with other forwards: broadcast collapsed coeffs across lags
            coeffs_f_broadcast = coeffs_f_collapsed[:, None, :, :].expand(-1, O, -1, -1)  # (Bch,O,P,P)

            preds_out.append(preds_f)
            coeffs_time_out.append(coeffs_f_broadcast)   # treat as "time-like"
            coeffs_freq_out.append(coeffs_f_seq)

        preds = torch.cat(preds_out, dim=0)                  # (B, P)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0) # (B, O, P, P)
        coeffs_freq_seq = torch.cat(coeffs_freq_out, dim=0)  # (B, F, P, P)

        return preds, coeffs_time_like, coeffs_freq_seq


    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        if self.combine_method == "gated":
            return self.forward_gated(inputs, batch_chunk_size)
        elif self.combine_method == "sum":
            return self.forward_sum(inputs, batch_chunk_size)
        elif self.combine_method == "concat":
            return self.forward_concat(inputs, batch_chunk_size)
        elif self.combine_method == "freq_only":
            return self.forward_freq_only(inputs, batch_chunk_size)

import math

class ConditionalDiffusion(nn.Module):
    def __init__(self, num_vars, hidden_dim=256, timesteps=1000, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps
        self.device = device

        # Linear beta schedule
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # Timestep embedding (sinusoidal)
        self.timestep_embed = nn.Embedding(timesteps, hidden_dim)

        # Denoiser network (conditioned on coeffs)
        self.denoise_net = nn.Sequential(
            nn.Linear(num_vars + num_vars * num_vars + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_vars)  # predict noise
        )

    def forward_diffusion(self, x0, t):
        """
        Sample forward process: q(x_t | x0)
        """
        sqrt_alpha_cum = torch.sqrt(self.alphas_cumprod[t]).unsqueeze(-1)
        sqrt_one_minus = torch.sqrt(1 - self.alphas_cumprod[t]).unsqueeze(-1)
        noise = torch.randn_like(x0)
        xt = sqrt_alpha_cum * x0 + sqrt_one_minus * noise
        return xt, noise

    def predict_noise(self, xt, coeffs, t):
        """
        Predict noise eps ~ p_theta(eps | xt, coeffs, t)
        """
        t_embed = self.timestep_embed(t)
        cond = torch.cat([xt, coeffs.view(coeffs.size(0), -1), t_embed], dim=-1)
        eps_hat = self.denoise_net(cond)
        return eps_hat

    @torch.no_grad()
    def sample(self, coeffs, shape):
        """
        Generate a prediction from noise, conditioned on coeffs
        """
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps_hat = self.predict_noise(x, coeffs, t_tensor)

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_cum = self.alphas_cumprod[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cum)) * eps_hat) + torch.sqrt(beta_t) * noise
        return x


class RecurrentAttentionGNN_Attn_crossattn(nn.Module):
    """
    Time domain path: like original RecurrentAttentionGNN_Attn.
    Freq domain path: rFFT over lags (per variable) -> magnitude -> per-bin GNN -> attention over bins.
    Cross-attention: time <-> freq interaction before prediction.
    """
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01, options=None):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale
        self.dynamic_gating = False
        self.combine_coeffs = "None"  # if False, just return time & freq coeffs separately
        self.diffusion_for_pred = False 
        self.time_freq_representation = options.get("time_freq_representation", "") # normal, mag_phase, learnable_filter
        # Shared GNN per slice
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # --- Time path ---
        self.in_proj_time = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_time = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_time = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

        # --- Freq path ---
        self.in_proj_freq = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_freq = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_freq = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_freq = nn.Parameter(torch.randn(1, (order // 2) + 1, hidden_dim) * pe_scale)

        # --- Cross-attention ---
        self.cross_attn_time_to_freq = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_freq_to_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # --- Dynamic gating ---
        if self.dynamic_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        if self.combine_coeffs == "cnn":
            # Fuse time & freq coefficients using a small CNN
            # Input channels: time coeffs + freq coeffs (stack along channel)
            # Output channels: keep same as number of lags (O) for prediction
            self.coeff_fusion_cnn = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),  # fuse time+freq for each lag
                nn.ReLU(),
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
                nn.ReLU()
            )
        if self.diffusion_for_pred:
            # Use a 1d diffusion model to refine predictions with coeff as conditioning
            self.diffusion_model = ConditionalDiffusion(
                num_vars=num_vars,
                hidden_dim=hidden_dim,
                timesteps=1000,
                device=device
            )

        if self.time_freq_representation == "mag_phase":
            self.in_projector = nn.Linear(num_vars*2, num_vars)
        elif self.time_freq_representation == "mag_phase_learnable_filter":
            self.in_projector = nn.Linear(num_vars*2, num_vars)
            self.texfilter = TexFilter(
                embed_size=num_vars,
                use_gelu=True,             # or use_swish=True for smoother nonlinearity
                use_skip=True,             # ✅ Preserve original signal paths
                use_layernorm=True,        # ✅ Stabilize across frequency bins
                hard_threshold=False,      # ❌ Avoid hard cutting off weak signals
                use_window=False,          # ❌ Avoid muting boundary info
                sparsity_threshold=0.0     # ✅ Retain all weak signal components
            )
        elif self.time_freq_representation == "learnable_filter":
            self.texfilter = TexFilter(
                embed_size=num_vars,
                use_gelu=True,             # or use_swish=True for smoother nonlinearity
                use_skip=True,             # ✅ Preserve original signal paths
                use_layernorm=True,        # ✅ Stabilize across frequency bins
                hard_threshold=False,      # ❌ Avoid hard cutting off weak signals
                use_window=False,          # ❌ Avoid muting boundary info
                sparsity_threshold=0.0     # ✅ Retain all weak signal components
            )

            
    def _augment_graph(self, coeffs, node_mask_prob=0.1, edge_drop_prob=0.1):
        """On-the-fly augmentation for loss computation."""
        B, O, P, _ = coeffs.shape
        aug = coeffs.clone()

        if node_mask_prob > 0:
            node_mask = (torch.rand(B, 1, P, 1, device=coeffs.device) > node_mask_prob).float()
            aug = aug * node_mask
        if edge_drop_prob > 0:
            edge_mask = (torch.rand(B, O, P, P, device=coeffs.device) > edge_drop_prob).float()
            aug = aug * edge_mask
        return aug

    def _init_weights(self):
        """Initialize weights for linear layers and attention projections."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                # PyTorch MHA already uses xavier for in_proj_weight internally, but can re-init if needed
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
                    
    @torch.no_grad()
    def _context_stats(self, x_win):  # (B, O, P)
        mean = x_win.mean(dim=1)      # (B, P)
        std = x_win.std(dim=1)        # (B, P)
        return torch.cat([mean, std], dim=-1)  # (B, 2P)

    def _time_path(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape
        feats = []
        for k in range(O):
            coeff_k, _ = self.base_net(x_chunk[:, k, :])  # (Bc, P, P)
            feats.append(coeff_k.view(Bc, -1))            # (Bc, P*P)
        seq = torch.stack(feats, dim=1)                   # (Bc, O, P*P)
        attn_in = self.in_proj_time(seq) + self.pos_enc_time[:, :O, :]
        attn_out, _ = self.temporal_attn_time(attn_in, attn_in, attn_in)  # (Bc, O, H)
        return attn_out, seq  # return seq for potential diagnostics

    def _freq_path_normal(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape
        X = torch.fft.rfft(x_chunk, dim=1)             # (Bc, F, P) complex
        Xmag = X.abs()                                 # (Bc, F, P)
        Fbins = Xmag.size(1)

        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(Xmag[:, f, :])  # (Bc, P, P)
            feats.append(coeff_f.view(Bc, -1))         # (Bc, P*P)
        seq = torch.stack(feats, dim=1)                # (Bc, F, P*P)
        attn_in = self.in_proj_freq(seq) + self.pos_enc_freq[:, :Fbins, :]
        attn_out, _ = self.temporal_attn_freq(attn_in, attn_in, attn_in)  # (Bc, F, H)
        return attn_out, seq, Xmag

    def _freq_path_mag_phase(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape
        # --- 1. rFFT ---
        X = torch.fft.rfft(x_chunk, dim=1)           # (Bc, F, P) complex
        # --- 2. Extract magnitude and phase ---
        X_mag = X.abs()                              # (Bc, F, P)
        X_phase = torch.angle(X)                     # (Bc, F, P)
        X_phase_norm = X_phase / np.pi               # now in [-1, 1]
        # --- 3. Concatenate to preserve full signal ---
        X_full = torch.cat([X_mag, X_phase_norm], dim=-1)  # (Bc, F, 2*P)
        Fbins = X_full.size(1)
        # --- 4. GNN per frequency bin ---
        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(self.in_projector(X_full[:, f, :]))  # (Bc, P, P)
            feats.append(coeff_f.view(Bc, -1))           # flatten to (Bc, P*P)
        seq = torch.stack(feats, dim=1)                  # (Bc, F,      P*P)            
        # --- 5. Temporal attention over frequency bins ---
        attn_in = self.in_proj_freq(seq) + self.pos_enc_freq[:, :Fbins, :]
        attn_out, _ = self.temporal_attn_freq(attn_in,
                                                attn_in, attn_in)   # (Bc, F, H)
        return attn_out, seq, X_full
    
    def _freq_path_mag_phase_learnable_filter(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape

        # --- 1. rFFT ---
        X = torch.fft.rfft(x_chunk, dim=1)           # (Bc, F, P) complex

        # --- 2. Apply learnable frequency filter (TexFilter) ---
        X = X * self.texfilter(X)                    # element-wise frequency attention

        # --- 3. Extract real and imaginary channels ---
        X_real = X.real                              # (Bc, F, P)
        X_imag = X.imag                              # (Bc, F, P)

        # --- 4. Concatenate to preserve full signal ---
        X_full = torch.cat([X_real, X_imag], dim=-1)  # (Bc, F, 2*P)

        Fbins = X_full.size(1)

        # --- 5. GNN per frequency bin ---
        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(self.in_projector(X_full[:, f, :])) # (Bc, P, P)
            feats.append(coeff_f.view(Bc, -1))           # flatten to (Bc, P*P)
        seq = torch.stack(feats, dim=1)                  # (Bc, F, P*P)

        # --- 6. Temporal attention over frequency bins ---
        attn_in = self.in_proj_freq(seq) + self.pos_enc_freq[:, :Fbins, :]
        attn_out, _ = self.temporal_attn_freq(attn_in, attn_in, attn_in)  # (Bc, F, H)

        return attn_out, seq, X_full

    def _freq_path_learnable_filter(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape

        # --- 1. rFFT ---
        X = torch.fft.rfft(x_chunk, dim=1)           # (Bc, F, P) complex

        # --- 2. Apply learnable frequency filter (TexFilter) ---
        X = X * self.texfilter(X)                    # element-wise frequency attention

        # --- 3. Extract real and imaginary channels ---
        X_real = X.abs()                             # (Bc, F, P)

        Fbins = X_real.size(1)

        # --- 5. GNN per frequency bin ---
        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(X_real[:, f, :]) # (Bc, P, P)
            feats.append(coeff_f.view(Bc, -1))           # flatten to (Bc, P*P)
        seq = torch.stack(feats, dim=1)                  # (Bc, F, P*P)

        # --- 6. Temporal attention over frequency bins ---
        attn_in = self.in_proj_freq(seq) + self.pos_enc_freq[:, :Fbins, :]
        attn_out, _ = self.temporal_attn_freq(attn_in, attn_in, attn_in)  # (Bc, F, H)

        return attn_out, seq, X_real

    def _freq_path(self, x_chunk):  # x_chunk: (B_chunk, O, P)
        if self.time_freq_representation == "normal":
            return self._freq_path_normal(x_chunk)
        elif self.time_freq_representation == "mag_phase":
            return self._freq_path_mag_phase(x_chunk)
        elif self.time_freq_representation == "mag_phase_learnable_filter":
            return self._freq_path_mag_phase_learnable_filter(x_chunk)
        elif self.time_freq_representation == "learnable_filter":
            # same as mag_phase_learnable_filter but without phase extraction step
            return self._freq_path_learnable_filter(x_chunk)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape

        preds_out, coeffs_time_out, coeffs_freq_out = [], [], []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)

            # --- encode time & freq ---
            attn_time, _ = self._time_path(x_chunk)  # (Bch, O, H)
            attn_freq, _, _ = self._freq_path(x_chunk)  # (Bch, F, H)

            # --- cross-attention ---
            attn_time_cross, _ = self.cross_attn_time_to_freq(attn_time, attn_freq, attn_freq)  # (Bch, O, H)
            attn_freq_cross, _ = self.cross_attn_freq_to_time(attn_freq, attn_time, attn_time)  # (Bch, F, H)

            # --- project to coeffs ---
            coeffs_time_seq = self.coeff_proj_time(attn_time_cross)  # (Bch, O, P*P)
            coeffs_time_seq = coeffs_time_seq.view(-1, O, P, P)

            coeffs_freq_seq = self.coeff_proj_freq(attn_freq_cross)  # (Bch, F, P*P)
            Fbins = coeffs_freq_seq.size(1)
            coeffs_freq_seq = coeffs_freq_seq.view(-1, Fbins, P, P)
            coeffs_freq_collapsed = coeffs_freq_seq.mean(dim=1)     # (Bch, P, P)

            # --- prediction ---
            #if self.diffusion_for_pred:
            #    # Use time coeffs as condition
            #    coeffs_flat = coeffs_time_seq.view(x_chunk.size(0), -1)  # (B, O*P*P)
#
            #    # Sample prediction from diffusion conditioned on coeffs
            #    preds_time = self.diffusion_model.sample(coeffs_flat, (x_chunk.size(0), P))
#
            #    # Still keep freq prediction (linear)
            #    preds_freq = (coeffs_freq_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)
#
            #else:
            preds_time = torch.zeros((x_chunk.size(0), P), device=x_chunk.device)
            for k in range(O):
                preds_time += (coeffs_time_seq[:, k] @ x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)
            preds_freq = (coeffs_freq_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)

            # --- final fused prediction ---
            if not self.dynamic_gating:
                preds = 0.5 * preds_time + 0.5 * preds_freq
            else:
                gate_input = torch.cat([attn_time_cross.mean(dim=1), attn_freq_cross.mean(dim=1)], dim=-1)
                gate = torch.sigmoid(self.gate_net(gate_input))
                preds = gate * preds_time + (1 - gate) * preds_freq
            preds_out.append(preds)
            coeffs_time_out.append(coeffs_time_seq)
            coeffs_freq_out.append(coeffs_freq_seq)

            # --- Fuse coeffs (time + freq) ---
            if self.combine_coeffs == "weighted_sum":
                alpha_time = torch.tensor(0.5, device=coeffs_time_seq.device)

                alpha_freq = 1.0 - alpha_time  # same shape, broadcastable

                # expand freq coefficients along lag dimension
                coeffs_freq_exp = coeffs_freq_collapsed[:, None, :, :]  # (Bch, 1, P, P)

                # fused coefficients
                coeffs_fused = alpha_time * coeffs_time_seq + alpha_freq * coeffs_freq_exp

                # store fused version
                coeffs_time_out[-1] = coeffs_fused
                coeffs_freq_out[-1] = coeffs_freq_collapsed  # keep collapsed for reference
            elif self.combine_coeffs == "cnn":
                # Expand freq coefficients along lag dimension
                coeffs_freq_exp = coeffs_freq_collapsed[:, None, :, :]  # (Bch, 1, P, P)

                # Stack time & freq along channel dimension
                coeffs_stack = torch.stack([coeffs_time_seq, coeffs_freq_exp.repeat(1, O, 1, 1)], dim=2)
                # coeffs_stack shape: (Bch, O, 2, P, P)

                Bch, O, C, P, _ = coeffs_stack.shape
                coeffs_stack = coeffs_stack.view(Bch*O, C, P, P)  # treat each lag separately as batch

                # Apply CNN fusion
                coeffs_fused = self.coeff_fusion_cnn(coeffs_stack)  # (Bch*O, 1, P, P)
                coeffs_fused = coeffs_fused.view(Bch, O, P, P)      # reshape back

                # Store fused version
                coeffs_time_out[-1] = coeffs_fused
                coeffs_freq_out[-1] = coeffs_freq_collapsed  # keep for reference

        preds = torch.cat(preds_out, dim=0)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0)
        coeffs_freq_seq = torch.cat(coeffs_freq_out, dim=0)

        return preds, coeffs_time_like, coeffs_freq_seq


class RecurrentAttentionGNN_Attn_crossattn_enhanced(nn.Module):
    """
    Enhanced RecurrentAttentionGNN with:
    - Time & frequency paths
    - Cross-attention (time <-> freq)
    - Variable-wise attention
    - Dynamic gating for fusion
    - Residual connections
    - Coefficient sparsity regularization
    """
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2,
                 attention_heads=4, attention_dim=64, pe_scale=0.01, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale
        self.dynamic_gating = False

        # Shared coefficient generator
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # --- Time path ---
        self.in_proj_time = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_time = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_time = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)
        self.var_attn_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # --- Frequency path ---
        self.in_freq_to_base = nn.Linear(num_vars*2, num_vars)  # to match base_net input dim
        self.in_proj_freq = nn.Linear(num_vars * num_vars, hidden_dim)  # complex: real + imag
        self.temporal_attn_freq = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_freq = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_freq = nn.Parameter(torch.randn(1, (order // 2) + 1, hidden_dim) * pe_scale)
        self.var_attn_freq = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # --- Cross-attention ---
        self.cross_attn_time_to_freq = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_freq_to_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # --- Dynamic gating ---
        if self.dynamic_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def _init_weights(self):
        """Initialize weights for linear layers and attention projections."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                # PyTorch MHA already uses xavier for in_proj_weight internally, but can re-init if needed
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
    
    @torch.no_grad()
    def _context_stats(self, x_win):  # (B, O, P)
        mean = x_win.mean(dim=1)      # (B, P)
        std = x_win.std(dim=1)        # (B, P)
        return torch.cat([mean, std], dim=-1)  # (B, 2P)

    def _time_path(self, x_chunk):
        Bc, O, P = x_chunk.shape
        feats = []
        for k in range(O):
            coeff_k, _ = self.base_net(x_chunk[:, k, :])
            feats.append(coeff_k.view(Bc, -1))
        seq = torch.stack(feats, dim=1)
        attn_in = self.in_proj_time(seq) + self.pos_enc_time[:, :O, :]
        attn_out, _ = self.temporal_attn_time(attn_in, attn_in, attn_in)
        attn_out, _ = self.var_attn_time(attn_out, attn_out, attn_out)
        return attn_out, seq

    def _freq_path(self, x_chunk):
        Bc, O, P = x_chunk.shape
        X = torch.fft.rfft(x_chunk, dim=1)
        X_complex = torch.cat([X.real, X.imag], dim=-1)  # include phase info
        Fbins = X_complex.size(1)
        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(self.in_freq_to_base(X_complex[:, f, :]))
            feats.append(coeff_f.view(Bc, -1))
        seq = torch.stack(feats, dim=1)
        attn_in = self.in_proj_freq(seq) + self.pos_enc_freq[:, :Fbins, :]
        attn_out, _ = self.temporal_attn_freq(attn_in, attn_in, attn_in)
        attn_out, _ = self.var_attn_freq(attn_out, attn_out, attn_out)
        return attn_out, seq, X_complex

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape
        preds_out, coeffs_time_out, coeffs_freq_out = [], [], []
        attn_time_weights, attn_freq_weights = [], []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]

            # Encode paths
            attn_time, _ = self._time_path(x_chunk)
            attn_freq, _, _ = self._freq_path(x_chunk)

            # Cross-attention with residual
            attn_time_cross, _ = self.cross_attn_time_to_freq(attn_time, attn_freq, attn_freq)
            attn_time_cross += attn_time

            attn_freq_cross, _ = self.cross_attn_freq_to_time(attn_freq, attn_time, attn_time)
            attn_freq_cross += attn_freq

            # Project to coefficients
            coeffs_time_seq = self.coeff_proj_time(attn_time_cross).view(-1, O, P, P)
            coeffs_freq_seq = self.coeff_proj_freq(attn_freq_cross).view(-1, attn_freq_cross.size(1), P, P)
            coeffs_freq_collapsed = coeffs_freq_seq.mean(dim=1)

            # Predictions
            preds_time = sum((coeffs_time_seq[:, k] @ x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)
                             for k in range(O))
            preds_freq = (coeffs_freq_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)

            # Dynamic gating
            if not self.dynamic_gating:
                preds = 0.5 * preds_time + 0.5 * preds_freq
            else:
                gate_input = torch.cat([attn_time_cross.mean(dim=1), attn_freq_cross.mean(dim=1)], dim=-1)
                gate = torch.sigmoid(self.gate_net(gate_input))
                preds = gate * preds_time + (1 - gate) * preds_freq

            preds_out.append(preds)
            coeffs_time_out.append(coeffs_time_seq)
            coeffs_freq_out.append(coeffs_freq_seq)
            attn_time_weights.append(attn_time_cross)
            attn_freq_weights.append(attn_freq_cross)

        preds = torch.cat(preds_out, dim=0)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0)
        coeffs_freq_seq = torch.cat(coeffs_freq_out, dim=0)
        attn_time_weights = torch.cat(attn_time_weights, dim=0)
        attn_freq_weights = torch.cat(attn_freq_weights, dim=0)

        return preds, coeffs_time_like, coeffs_freq_seq#, attn_time_weights, attn_freq_weights

    def coefficient_sparsity_loss(self, coeffs_time, coeffs_freq):
        loss_time = (coeffs_time.softmax(dim=-1) * torch.log(coeffs_time.softmax(dim=-1) + 1e-8)).sum()
        loss_freq = (coeffs_freq.softmax(dim=-1) * torch.log(coeffs_freq.softmax(dim=-1) + 1e-8)).sum()
        return loss_time + loss_freq

import torch
import torch.nn as nn
import numpy as np
from numpy.polynomial import Legendre as L

def leg_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1
    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')
    # Make sure data is on the right device
    data = data.to(device)
    tvals = np.linspace(-1, 1, T)
    legendre_polys = np.array([L.basis(i)(tvals) for i in range(degree)])
    legendre_polys = torch.from_numpy(legendre_polys).float().to(device)  # shape: [degree, T]
    # Compute coefficients
    coeffs_candidate = torch.mm(legendre_polys, data) / T * 2
    coeffs = torch.stack([coeffs_candidate[i] * (2 * i + 1) / 2 for i in range(degree)]).to(device)
    coeffs = coeffs.transpose(0, 1)  # shape: [B * D, degree]
    if rtn_data:
        reconstructed_data = torch.mm(coeffs, legendre_polys)
        reconstructed_data = reconstructed_data.reshape(B, -1, T).permute(0, 2, 1)
        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs

def legendre_encode(input_seq, degree=64):
    B, T, C = input_seq.shape
    input_seq_flat = input_seq.reshape(B * C, T).T  # shape: [T, B*C]
    device = input_seq.device
    coeffs = leg_torch(input_seq_flat, degree - 1, rtn_data=False, device=device)
    coeffs = coeffs.reshape(B, C, degree).to(device)
    return coeffs

def legendre_decode(coeffs, seq_len):
    B, C, D = coeffs.shape
    coeffs_flat = coeffs.reshape(B * C, D)
    # Generate Legendre basis on correct device
    tvals = np.linspace(-1, 1, seq_len)
    legendre_polys = np.array([L.basis(i)(tvals) for i in range(D)])  # [D, T]
    legendre_polys = torch.from_numpy(legendre_polys).float().to(coeffs.device)
    reconstructed = torch.mm(coeffs_flat, legendre_polys).reshape(B, C, seq_len).permute(0, 2, 1)
    return reconstructed

class RecurrentAttentionGNN_Attn_crossattn_Legendre(nn.Module):
    """
    Time domain path: like original RecurrentAttentionGNN_Attn.
    Legendre domain path: efficient Legendre transform -> coefficients -> per-coeff GNN -> attention over coefficients.
    Cross-attention: time <-> legendre interaction before prediction.
    """
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01, legendre_degree=None):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale
        
        # Set Legendre polynomial degree
        self.legendre_degree = legendre_degree if legendre_degree is not None else min(order, 32)

        # Shared GNN per slice
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # --- Time path ---
        self.in_proj_time = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_time = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_time = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

        # --- Legendre path ---
        self.in_proj_legendre = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_legendre = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_legendre = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_legendre = nn.Parameter(torch.randn(1, self.legendre_degree, hidden_dim) * pe_scale)

        # --- Cross-attention ---
        self.cross_attn_time_to_legendre = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_legendre_to_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def _init_weights(self):
        """Initialize weights for linear layers and attention projections."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

    @torch.no_grad()
    def _context_stats(self, x_win):  # (B, O, P)
        mean = x_win.mean(dim=1)      # (B, P)
        std = x_win.std(dim=1)        # (B, P)
        return torch.cat([mean, std], dim=-1)  # (B, 2P)

    def _time_path(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape
        feats = []
        for k in range(O):
            coeff_k, _ = self.base_net(x_chunk[:, k, :])  # (Bc, P, P)
            feats.append(coeff_k.view(Bc, -1))            # (Bc, P*P)
        seq = torch.stack(feats, dim=1)                   # (Bc, O, P*P)
        attn_in = self.in_proj_time(seq) + self.pos_enc_time[:, :O, :]
        attn_out, _ = self.temporal_attn_time(attn_in, attn_in, attn_in)  # (Bc, O, H)
        return attn_out, seq

    def _legendre_path(self, x_chunk):  # (B_chunk, O, P)
        Bc, O, P = x_chunk.shape
        
        # Use the efficient Legendre encoding
        legendre_coeffs = legendre_encode(x_chunk, degree=self.legendre_degree)  # (Bc, P, degree)
        # Transpose to get coefficients as sequence dimension: (Bc, degree, P)
        legendre_coeffs = legendre_coeffs.transpose(1, 2)
        
        feats = []
        for c in range(self.legendre_degree):
            coeff_c, _ = self.base_net(legendre_coeffs[:, c, :])  # (Bc, P, P)
            feats.append(coeff_c.view(Bc, -1))                    # (Bc, P*P)
        seq = torch.stack(feats, dim=1)                           # (Bc, degree, P*P)
        
        attn_in = self.in_proj_legendre(seq) + self.pos_enc_legendre
        attn_out, _ = self.temporal_attn_legendre(attn_in, attn_in, attn_in)  # (Bc, degree, H)
        
        return attn_out, seq, legendre_coeffs

    def _reconstruct_from_legendre(self, legendre_coeffs, target_seq_len):
        """
        Reconstruct time series from Legendre coefficients at target sequence length.
        Args:
            legendre_coeffs: (B_chunk, degree, P)
            target_seq_len: length of sequence to reconstruct
        Returns:
            reconstructed: (B_chunk, target_seq_len, P)
        """
        # Transpose to match expected format for decode function: (Bc, P, degree)
        coeffs_for_decode = legendre_coeffs.transpose(1, 2)
        
        # Reconstruct full sequence
        reconstructed_seq = legendre_decode(coeffs_for_decode, target_seq_len)  # (Bc, target_seq_len, P)
        
        return reconstructed_seq

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape

        preds_out, coeffs_time_out, coeffs_legendre_out = [], [], []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)

            # --- encode time & legendre ---
            attn_time, _ = self._time_path(x_chunk)  # (Bch, O, H)
            attn_legendre, _, legendre_coeffs = self._legendre_path(x_chunk)  # (Bch, degree, H)

            # --- cross-attention ---
            attn_time_cross, _ = self.cross_attn_time_to_legendre(attn_time, attn_legendre, attn_legendre)  # (Bch, O, H)
            attn_legendre_cross, _ = self.cross_attn_legendre_to_time(attn_legendre, attn_time, attn_time)  # (Bch, degree, H)

            # --- project to coeffs ---
            coeffs_time_seq = self.coeff_proj_time(attn_time_cross)  # (Bch, O, P*P)
            coeffs_time_seq = coeffs_time_seq.view(-1, O, P, P)

            coeffs_legendre_seq = self.coeff_proj_legendre(attn_legendre_cross)  # (Bch, degree, P*P)
            coeffs_legendre_seq = coeffs_legendre_seq.view(-1, self.legendre_degree, P, P)
            coeffs_legendre_collapsed = coeffs_legendre_seq.mean(dim=1)     # (Bch, P, P)

            # --- prediction ---
            # Time domain prediction
            preds_time = torch.zeros((x_chunk.size(0), P), device=x_chunk.device)
            for k in range(O):
                # coeffs_time_seq[:, k] is (Bch, P, P), x_chunk[:, k, :] is (Bch, P)
                preds_time += (coeffs_time_seq[:, k] @ x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)
            
            # Legendre domain prediction - reconstruct at last time point (single value)
            reconstructed_seq = self._reconstruct_from_legendre(legendre_coeffs, O)  # (Bch, O, P)
            legendre_last_point = reconstructed_seq[:, -1, :]  # (Bch, P) - last time point
            preds_legendre = (coeffs_legendre_collapsed @ legendre_last_point.unsqueeze(-1)).squeeze(-1)

            # --- final fused prediction ---
            preds = 0.5 * preds_time + 0.5 * preds_legendre  # can adjust weighting or use a learned gate

            preds_out.append(preds)
            coeffs_time_out.append(coeffs_time_seq)
            coeffs_legendre_out.append(coeffs_legendre_seq)

        preds = torch.cat(preds_out, dim=0)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0)
        coeffs_legendre_seq = torch.cat(coeffs_legendre_out, dim=0)

        return preds, coeffs_time_like, coeffs_legendre_seq



class RecurrentAttentionGNN_Attn_Enhanced(nn.Module):
    """
    Enhanced version with:
    1. Learnable fusion gate instead of fixed 0.5 weighting
    2. Multi-scale frequency analysis
    3. Adaptive positional encoding
    4. Residual connections and layer normalization
    5. Frequency-aware attention with phase information
    6. Dynamic coefficient regularization
    7. Hierarchical attention (variable-level + temporal)
    8. Optional causal masking for online prediction
    """
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01, dropout=0.1,
                 use_phase=True, multi_scale_freqs=[1, 2, 4], use_causal_mask=False,
                 regularization_strength=0.01):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale
        self.use_phase = use_phase
        self.multi_scale_freqs = multi_scale_freqs
        self.use_causal_mask = use_causal_mask
        self.reg_strength = regularization_strength

        # Shared GNN per slice
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # --- Enhanced Time Path ---
        self.in_proj_time = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_time = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.layer_norm_time = nn.LayerNorm(hidden_dim)
        self.coeff_proj_time = nn.Linear(hidden_dim, num_vars * num_vars)
        
        # Adaptive positional encoding
        self.pos_enc_time = AdaptivePositionalEncoding(hidden_dim, max_len=order)

        # --- Enhanced Freq Path with Multi-scale ---
        freq_input_dim = (num_vars * num_vars) * len(multi_scale_freqs)
        if use_phase:
            freq_input_dim *= 2  # magnitude + phase
            
        self.in_proj_freq = nn.Linear(freq_input_dim, hidden_dim)
        self.temporal_attn_freq = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.layer_norm_freq = nn.LayerNorm(hidden_dim)
        self.coeff_proj_freq = nn.Linear(hidden_dim, num_vars * num_vars)
        
        # Dynamic positional encoding based on frequency bins
        self.freq_pos_embedding = nn.Embedding(order + 1, hidden_dim)

        # --- Enhanced Cross-attention with Gating ---
        self.cross_attn_time_to_freq = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.cross_attn_freq_to_time = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        
        # Learnable fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

        # --- Hierarchical Variable Attention ---
        # Project variables to a suitable embedding dimension first
        self.var_embed_dim = min(hidden_dim, 64)  # Reasonable embedding size
        self.var_input_proj = nn.Linear(num_vars, self.var_embed_dim)
        self.var_output_proj = nn.Linear(self.var_embed_dim, num_vars)
        self.var_attention = nn.MultiheadAttention(
            embed_dim=self.var_embed_dim, 
            num_heads=min(num_heads, self.var_embed_dim // 16), 
            batch_first=True
        )

        # --- Regularization components ---
        self.dropout = nn.Dropout(dropout)
        self.coeff_regularizer = CoefficientRegularizer(regularization_strength)
        
        self._init_weights()

    def _init_weights(self):
        """Enhanced weight initialization with proper scaling."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'gate' in name:
                    # Initialize gate to favor balanced fusion initially
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

    def _multi_scale_freq_analysis(self, x_chunk):
        """Multi-scale frequency analysis with optional phase information."""
        Bc, O, P = x_chunk.shape
        all_freq_features = []
        
        for scale in self.multi_scale_freqs:
            # Downsample or use sliding windows for different scales
            if scale == 1:
                x_scaled = x_chunk
            else:
                x_scaled = x_chunk[:, ::scale, :]  # Simple downsampling
                if x_scaled.size(1) < 3:  # Ensure minimum length
                    x_scaled = x_chunk[:, -3:, :]
            
            X = torch.fft.rfft(x_scaled, dim=1)
            Xmag = X.abs()
            
            if self.use_phase:
                Xphase = X.angle()
                X_combined = torch.cat([Xmag, Xphase], dim=-1)  # (Bc, F, 2P)
            else:
                X_combined = Xmag  # (Bc, F, P)
            
            Fbins = X_combined.size(1)
            scale_features = []
            
            for f in range(Fbins):
                if self.use_phase:
                    # Split back into mag and phase for GNN processing
                    mag_part = X_combined[:, f, :P]
                    phase_part = X_combined[:, f, P:]
                    # Use magnitude for GNN, incorporate phase as auxiliary feature
                    coeff_f, _ = self.base_net(mag_part)
                    # Phase-modulated coefficients
                    phase_weight = torch.cos(phase_part).mean(dim=-1, keepdim=True).unsqueeze(-1)
                    coeff_f = coeff_f * (1 + 0.1 * phase_weight)  # Subtle phase influence
                else:
                    coeff_f, _ = self.base_net(X_combined[:, f, :])
                
                scale_features.append(coeff_f.view(Bc, -1))
            
            if scale_features:
                scale_seq = torch.stack(scale_features, dim=1)  # (Bc, F, P*P)
                all_freq_features.append(scale_seq)
        
        # Concatenate multi-scale features
        if all_freq_features:
            # Pad sequences to same length and concatenate
            max_len = max(feat.size(1) for feat in all_freq_features)
            padded_features = []
            for feat in all_freq_features:
                if feat.size(1) < max_len:
                    pad_size = max_len - feat.size(1)
                    feat = torch.cat([feat, feat[:, -1:, :].repeat(1, pad_size, 1)], dim=1)
                padded_features.append(feat)
            
            freq_features = torch.cat(padded_features, dim=-1)  # Concat along feature dim
            return freq_features, max_len
        else:
            # Fallback to single scale
            return self._single_scale_freq_analysis(x_chunk)

    def _single_scale_freq_analysis(self, x_chunk):
        """Fallback single-scale analysis."""
        Bc, O, P = x_chunk.shape
        X = torch.fft.rfft(x_chunk, dim=1)
        Xmag = X.abs()
        Fbins = Xmag.size(1)
        
        feats = []
        for f in range(Fbins):
            coeff_f, _ = self.base_net(Xmag[:, f, :])
            feats.append(coeff_f.view(Bc, -1))
        
        seq = torch.stack(feats, dim=1)
        return seq, Fbins

    def _time_path(self, x_chunk):
        """Enhanced time path with residual connections and layer norm."""
        Bc, O, P = x_chunk.shape
        feats = []
        
        for k in range(O):
            coeff_k, attn_weights = self.base_net(x_chunk[:, k, :])
            feats.append(coeff_k.view(Bc, -1))
        
        seq = torch.stack(feats, dim=1)  # (Bc, O, P*P)
        
        # Input projection with residual
        proj_seq = self.in_proj_time(seq)
        pos_enc = self.pos_enc_time(torch.arange(O, device=seq.device))
        attn_in = proj_seq + pos_enc
        
        # Self-attention with residual and layer norm
        attn_out, attn_weights = self.temporal_attn_time(attn_in, attn_in, attn_in)
        attn_out = self.layer_norm_time(attn_out + proj_seq)
        attn_out = self.dropout(attn_out)
        
        return attn_out, seq, attn_weights

    def _freq_path(self, x_chunk):
        """Enhanced frequency path with multi-scale analysis."""
        freq_features, Fbins = self._single_scale_freq_analysis(x_chunk)
        Bc = freq_features.size(0)

        # Input projection
        proj_seq = self.in_proj_freq(freq_features)
        
        # Frequency-based positional encoding
        freq_indices = torch.arange(Fbins, device=freq_features.device)
        pos_enc = self.freq_pos_embedding(freq_indices).unsqueeze(0)
        attn_in = proj_seq + pos_enc
        
        # Self-attention with residual and layer norm
        attn_out, attn_weights = self.temporal_attn_freq(attn_in, attn_in, attn_in)
        attn_out = self.layer_norm_freq(attn_out + proj_seq)
        attn_out = self.dropout(attn_out)
        
        return attn_out, freq_features, attn_weights, Fbins

    def _hierarchical_variable_attention(self, x_chunk, time_coeffs, freq_coeffs):
        """Apply variable-level attention to coefficient matrices."""
        Bc, O, P, _ = time_coeffs.shape
        
        # Simple approach: treat each coefficient matrix as a sequence of variable vectors
        # Reshape: (Bc, O, P, P) -> (Bc*O, P, P)
        time_flat = time_coeffs.view(-1, P, P)  # (Bc*O, P, P)
        
        # Project each row (variable's coefficients) to embedding space
        # time_flat is (Bc*O, P, P) - we want to project the last dimension P -> var_embed_dim
        time_embedded = self.var_input_proj(time_flat)  # (Bc*O, P, var_embed_dim)
        
        # Now time_embedded is 3D: (Bc*O, P, var_embed_dim) ✓
        # This is exactly what MultiheadAttention expects!
        
        # Apply variable-level attention
        var_attn_out, var_attn_weights = self.var_attention(
            time_embedded, time_embedded, time_embedded
        )  # (Bc*O, P, var_embed_dim)
        
        # Project back to original space
        var_enhanced = self.var_output_proj(var_attn_out)  # (Bc*O, P, P)
        
        # Reshape back to original dimensions
        time_coeffs_enhanced = var_enhanced.view(Bc, O, P, P)
        
        return time_coeffs_enhanced

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 1000):
        B, O, P = inputs.shape
        
        preds_out, coeffs_time_out, coeffs_freq_out = [], [], []
        reg_loss = 0.0

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]
            
            # --- Enhanced time & freq encoding ---
            attn_time, seq_time, time_weights = self._time_path(x_chunk)
            attn_freq, seq_freq, freq_weights, Fbins = self._freq_path(x_chunk)
            
            # --- Cross-attention with residual connections ---
            attn_time_residual = attn_time
            attn_freq_residual = attn_freq
            
            attn_time_cross, _ = self.cross_attn_time_to_freq(
                attn_time, attn_freq, attn_freq
            )
            attn_freq_cross, _ = self.cross_attn_freq_to_time(
                attn_freq, attn_time, attn_time
            )
            
            # Residual connections
            attn_time_cross = attn_time_cross + attn_time_residual
            attn_freq_cross = attn_freq_cross + attn_freq_residual
            
            # --- Project to coefficients ---
            coeffs_time_seq = self.coeff_proj_time(attn_time_cross)
            coeffs_time_seq = coeffs_time_seq.view(-1, O, P, P)
            
            coeffs_freq_seq = self.coeff_proj_freq(attn_freq_cross)
            coeffs_freq_seq = coeffs_freq_seq.view(-1, Fbins, P, P)
            coeffs_freq_collapsed = coeffs_freq_seq.mean(dim=1)
            
            # --- Hierarchical variable attention ---
            coeffs_time_enhanced = self._hierarchical_variable_attention(
                x_chunk, coeffs_time_seq, coeffs_freq_collapsed
            )
            
            # --- Predictions ---
            preds_time = torch.zeros((x_chunk.size(0), P), device=x_chunk.device)
            for k in range(O):
                preds_time += (coeffs_time_enhanced[:, k] @ x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)
            
            preds_freq = (coeffs_freq_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)
            
            # --- Learnable fusion with context ---
            # Combine time and freq features for gate
            combined_features = torch.cat([
                attn_time_cross.mean(dim=1),  # Global time representation
                attn_freq_cross.mean(dim=1)   # Global freq representation
            ], dim=-1)
            
            fusion_weights = self.fusion_gate(combined_features)  # (Bc, 2)
            
            # Weighted combination
            preds = (fusion_weights[:, 0:1] * preds_time + 
                    fusion_weights[:, 1:2] * preds_freq)
            
            # --- Regularization ---
            reg_loss += self.coeff_regularizer(coeffs_time_enhanced, coeffs_freq_collapsed)
            
            preds_out.append(preds)
            coeffs_time_out.append(coeffs_time_enhanced)
            coeffs_freq_out.append(coeffs_freq_seq)

        final_preds = torch.cat(preds_out, dim=0)
        final_coeffs_time = torch.cat(coeffs_time_out, dim=0)
        final_coeffs_freq = torch.cat(coeffs_freq_out, dim=0)
        
        # Return predictions, coefficients, and regularization loss
        return final_preds, final_coeffs_time, final_coeffs_freq, reg_loss / len(preds_out)


class AdaptivePositionalEncoding(nn.Module):
    """Learnable positional encoding that adapts to sequence length."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable base encodings
        self.pe_base = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.length_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, positions):
        """
        Args:
            positions: tensor of position indices (seq_len,)
        Returns:
            positional encodings: (1, seq_len, d_model)
        """
        seq_len = len(positions)
        if seq_len <= self.max_len:
            return self.pe_base[:, :seq_len, :] * self.length_scale
        else:
            # Interpolate for longer sequences
            indices = torch.linspace(0, self.max_len-1, seq_len, device=positions.device).long()
            return self.pe_base[:, indices, :] * self.length_scale


class CoefficientRegularizer(nn.Module):
    """Regularizer for coefficient matrices to encourage stability."""
    def __init__(self, strength=0.01):
        super().__init__()
        self.strength = strength
    
    def forward(self, coeffs_time, coeffs_freq):
        """
        Apply regularization to coefficient matrices.
        Encourages:
        1. Spectral norm regularization for stability
        2. Sparsity in coefficients
        3. Similarity between time and freq coefficients
        """
        reg_loss = 0.0
        
        # Spectral norm regularization
        B = coeffs_time.size(0)
        for b in range(min(B, 10)):  # Limit for computational efficiency
            reg_loss += torch.norm(coeffs_time[b], p='fro') ** 2
            reg_loss += torch.norm(coeffs_freq[b], p='fro') ** 2
        
        # L1 sparsity
        reg_loss += torch.norm(coeffs_time, p=1)
        reg_loss += torch.norm(coeffs_freq, p=1)
        
        # Consistency between domains (optional)
        if coeffs_time.size() == coeffs_freq.size():
            reg_loss += torch.norm(coeffs_time - coeffs_freq, p='fro') ** 2
        
        return self.strength * reg_loss


# Additional utility for causal masking if needed
def create_causal_mask(seq_len, device):
    """Create causal mask for autoregressive attention."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask




class RecurrentAttentionGNN_Attn_legendre(nn.Module):
    """
    Time domain path: same as original.
    Freq domain path: Legendre polynomial projection over lags (using numpy) -> per-basis GNN -> attention over basis.
    Fusion: gated blend of time and polynomial coefficients.
    """
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01, num_basis=None):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale
        self.num_basis = num_basis or (order // 2 + 1)  # default similar to FFT bins

        # Shared per-slice GNN
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # --- Time path ---
        self.in_proj_time = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_time = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_time = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

        # --- Legendre path ---
        self.in_proj_legendre = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_attn_legendre = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.coeff_proj_legendre = nn.Linear(hidden_dim, num_vars * num_vars)
        self.pos_enc_legendre = nn.Parameter(torch.randn(1, self.num_basis, hidden_dim) * pe_scale)

        # --- Fusion gate ---
        ctx_dim = 2 * num_vars  # mean & std per variable
        self.fusion_gate = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Precompute Legendre basis using NumPy
        self.register_buffer("legendre_basis", self._compute_legendre_basis(order, self.num_basis))

    def _compute_legendre_basis(self, order, num_basis):
        """
        Generate Legendre polynomials using NumPy.
        Returns a tensor of shape (order, num_basis)
        """
        x = np.linspace(-1, 1, order)
        basis = []
        for n in range(num_basis):
            Pn = np.polynomial.legendre.Legendre([0]*n + [1])(x)  # n-th Legendre polynomial
            basis.append(Pn)
        basis = np.stack(basis, axis=1)  # (order, num_basis)
        return torch.tensor(basis, dtype=torch.float32)

    @torch.no_grad()
    def _context_stats(self, x_win):
        mean = x_win.mean(dim=1)
        std = x_win.std(dim=1)
        return torch.cat([mean, std], dim=-1)

    def _time_path(self, x_chunk):
        Bc, O, P = x_chunk.shape
        feats = []
        for k in range(O):
            coeff_k, _ = self.base_net(x_chunk[:, k, :])
            feats.append(coeff_k.view(Bc, -1))
        seq = torch.stack(feats, dim=1)
        attn_in = self.in_proj_time(seq) + self.pos_enc_time[:, :O, :]
        attn_out, _ = self.temporal_attn_time(attn_in, attn_in, attn_in)
        coeffs_seq = self.coeff_proj_time(attn_out).view(Bc, O, P, P)

        preds = torch.zeros((Bc, P), device=x_chunk.device)
        for k in range(O):
            preds += (coeffs_seq[:, k] @ x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)
        return preds, coeffs_seq

    def _legendre_path(self, x_chunk):
        # x_chunk: (B, O, P)
        Bc, O, P = x_chunk.shape
        order, num_basis = self.legendre_basis.shape

        # Resize legendre basis to match O
        legendre_resized = []
        for b in range(num_basis):
            leg_b = self.legendre_basis[:, b].cpu().numpy()           # shape (order,)
            leg_b_resized = np.interp(np.linspace(0, order-1, O), np.arange(order), leg_b)
            legendre_resized.append(leg_b_resized)
        legendre_resized = torch.tensor(np.stack(legendre_resized, axis=1), device=x_chunk.device, dtype=torch.float32)  # (O, num_basis)

        # Project input
        Xproj = torch.einsum('bok,kf->bfo', x_chunk, legendre_resized)

        feats = []
        for f in range(self.num_basis):
            coeff_f, _ = self.base_net(Xproj[:, f, :])
            feats.append(coeff_f.view(Bc, -1))
        seq = torch.stack(feats, dim=1)

        attn_in = self.in_proj_legendre(seq) + self.pos_enc_legendre[:, :self.num_basis, :]
        attn_out, _ = self.temporal_attn_legendre(attn_in, attn_in, attn_in)
        coeffs_seq = self.coeff_proj_legendre(attn_out).view(Bc, self.num_basis, P, P)

        coeffs_collapsed = coeffs_seq.mean(dim=1)
        preds = (coeffs_collapsed @ x_chunk[:, -1, :].unsqueeze(-1)).squeeze(-1)
        return preds, coeffs_seq, coeffs_collapsed


    def forward(self, inputs, batch_chunk_size=1000):
        B, O, P = inputs.shape
        preds_out, coeffs_time_out, coeffs_legendre_out = [], [], []
        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]

            preds_t, coeffs_t = self._time_path(x_chunk)
            preds_l, coeffs_l_seq, coeffs_l_collapsed = self._legendre_path(x_chunk)

            ctx = self._context_stats(x_chunk)
            alpha = self.fusion_gate(ctx)

            preds = alpha * preds_t + (1 - alpha) * preds_l

            alpha_time = alpha.view(-1, 1, 1, 1)
            alpha_leg = (1 - alpha).view(-1, 1, 1).unsqueeze(-1)
            coeffs_fused = alpha_time * coeffs_t + alpha_leg * coeffs_l_collapsed[:, None, :, :]

            preds_out.append(preds)
            coeffs_time_out.append(coeffs_fused)
            coeffs_legendre_out.append(coeffs_l_seq)

        preds = torch.cat(preds_out, dim=0)
        coeffs_time_like = torch.cat(coeffs_time_out, dim=0)
        coeffs_legendre_seq = torch.cat(coeffs_legendre_out, dim=0)

        return preds, coeffs_time_like, coeffs_legendre_seq

class RecurrentAttentionGNN_Attn_______(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=2, device="cpu",
                 attention_heads=4, attention_dim=64, pe_scale=0.01):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale

        # Base GNN per lag
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # Project GNN output to hidden_dim per lag
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)

        # Temporal attention across lags
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Residual connection for temporal attention
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Map attention hidden state back to coefficients
        self.coeff_proj = nn.Linear(hidden_dim, num_vars * num_vars)

        # Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 10000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list = []
        coeffs_list = []

        # Use positional encoding
        pos_embeddings = self.pos_enc[:, :O, :]  # shape (1, O, hidden_dim)

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]  # (B_chunk, O, P)
            B_chunk = x_chunk.size(0)

            # --- Step 1: extract features per lag ---
            features_seq = []
            for k in range(O):
                feat_k, _ = self.base_net(x_chunk[:, k, :])        # (B_chunk, P, P)
                # Flatten GNN output
                feat_k_flat = feat_k.view(B_chunk, -1)
                # Add input residual, tiled to match size
                feat_k_flat = feat_k_flat + x_chunk[:, k, :].repeat(1, P)
                features_seq.append(feat_k_flat)                   # (B_chunk, P*P)
            features_seq = torch.stack(features_seq, dim=1)       # (B_chunk, O, P*P)

            # --- Step 2: project to hidden_dim for attention + add positional encoding ---
            attn_input = self.in_proj(features_seq) + pos_embeddings  # (B_chunk, O, hidden_dim)

            # --- Step 3: temporal attention ---
            attn_out, _ = self.temporal_attn(attn_input, attn_input, attn_input)  # (B_chunk, O, hidden_dim)

            # --- Step 4: map attention outputs back to coeffs ---
            coeffs_seq = self.coeff_proj(attn_out)               # (B_chunk, O, P*P)
            coeffs_seq = coeffs_seq.view(B_chunk, O, P, P)       # (B_chunk, O, P, P)

            # --- Step 5: prediction using coeffs ---
            preds_chunk = torch.zeros((B_chunk, P), device=device)
            for k in range(O):
                preds_chunk += torch.matmul(coeffs_seq[:, k, :, :], 
                                            x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)

            preds_list.append(preds_chunk)
            coeffs_list.append(coeffs_seq)

        preds = torch.cat(preds_list, dim=0)                    # (B, P)
        coeffs = torch.cat(coeffs_list, dim=0)                  # (B, O, P, P)

        return preds, coeffs, None



class RecurrentAttentionGNN_Attn___(nn.Module):
    def __init__(self, num_vars, rank, order, hidden_dim=256, num_heads=4,
                 attention_heads=6, attention_dim=128, pe_scale=0.05, device="cpu"):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.order = order
        self.hidden_dim = hidden_dim
        self.device = device
        self.pe_scale = pe_scale

        # --- Base GNN per lag ---
        self.base_net = AttentionCoeffGNN_multihead_fixed(
            num_vars=num_vars, rank=rank, hidden_dim=attention_dim, heads=attention_heads
        )

        # --- Project GNN output to attention hidden_dim ---
        self.in_proj = nn.Linear(num_vars * num_vars, hidden_dim)

        # --- Learnable positional embedding ---
        self.pos_enc = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

        # --- Temporal multi-head attention (residual + normalization) ---
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_layer_norm = nn.LayerNorm(hidden_dim)

        # --- Dilated temporal convolution (optional, helps multi-scale) ---
        self.dilated_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)

        # --- Map attention output back to coefficients ---
        self.coeff_proj = nn.Linear(hidden_dim, num_vars * num_vars)

    def forward(self, inputs: torch.Tensor, batch_chunk_size: int = 10000):
        B, O, P = inputs.shape
        device = inputs.device

        preds_list, coeffs_list, attn_seq = [], [], []

        pos_embeddings = self.pos_enc[:, :O, :]

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = inputs[start:end]
            B_chunk = x_chunk.size(0)

            # --- Step 1: base GNN per lag ---
            features_seq = []
            attn_lags = []
            for k in range(O):
                feat_k, attn_k = self.base_net(x_chunk[:, k, :])
                features_seq.append(feat_k.view(B_chunk, -1))
                attn_lags.append(attn_k)
            features_seq = torch.stack(features_seq, dim=1)  # (B_chunk, O, P*P)
            attn_lags = torch.stack(attn_lags, dim=1)        # (B_chunk, O, P, P)

            # --- Step 2: project to hidden_dim + positional encoding ---
            attn_input = self.in_proj(features_seq) + pos_embeddings

            # --- Step 3: temporal attention with residual ---
            attn_out, _ = self.temporal_attn(attn_input, attn_input, attn_input)
            attn_out = self.attn_layer_norm(attn_out + attn_input)

            # --- Step 4: dilated temporal conv (multi-scale) ---
            attn_out = self.dilated_conv(attn_out.transpose(1, 2)).transpose(1, 2)

            # --- Step 5: map back to coefficient matrix ---
            coeffs_seq = self.coeff_proj(attn_out).view(B_chunk, O, P, P)

            # --- Step 6: prediction ---
            preds_chunk = torch.zeros(B_chunk, P, device=device)
            for k in range(O):
                preds_chunk += torch.matmul(coeffs_seq[:, k, :, :], x_chunk[:, k, :].unsqueeze(-1)).squeeze(-1)

            preds_list.append(preds_chunk)
            coeffs_list.append(coeffs_seq)
            attn_seq.append(attn_lags)

        preds = torch.cat(preds_list, dim=0)
        coeffs = torch.cat(coeffs_list, dim=0)
        attn_seq = torch.cat(attn_seq, dim=0)  # (B, O, P, P)

        return preds, coeffs, attn_seq


import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalGNN(nn.Module):
    def __init__(self, num_vars, rank, hidden_dim=64, heads=8, extra_layers=1, temporal_hidden=32):
        """
        TemporalGNN:
        - Spatial: AttentionCoeffGNN_multihead
        - Temporal: small GRUCell over spatial embeddings
        - Output: preds + coeffs
        """
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.temporal_hidden = temporal_hidden

        # Spatial GNN
        self.spatial_gnn = AttentionCoeffGNN_multihead(
            num_vars=num_vars,
            rank=rank,
            hidden_dim=hidden_dim,
            heads=heads,
            extra_layers=extra_layers
        )

        # Temporal projection (small recurrent model)
        self.proj = nn.Linear(num_vars * num_vars, hidden_dim)
        self.temporal_rnn = nn.GRUCell(hidden_dim, temporal_hidden)

        # Final MLP to produce U, V for coeffs
        self.final_mlp = nn.Sequential(
            nn.Linear(temporal_hidden, temporal_hidden),
            nn.ReLU(),
            nn.Linear(temporal_hidden, 2 * num_vars * rank)
        )

    def forward(self, x_seq):
        """
        x_seq: (B, order, num_vars)
        returns: preds, coeffs: (B, num_vars, num_vars)
        """
        B, order, p = x_seq.shape
        device = x_seq.device

        # Init temporal hidden state
        h_t = torch.zeros(B, self.temporal_hidden, device=device)

        for t in range(order):
            x_t = x_seq[:, t, :]  # (B, num_vars)

            # Spatial GNN produces coeffs
            coeffs_k = self.spatial_gnn(x_t)  # (B, p, p)

            # Flatten and project to hidden_dim
            h_embed = self.proj(coeffs_k.view(B, -1))  # (B, hidden_dim)

            # Update recurrent state
            h_t = self.temporal_rnn(h_embed, h_t)

        # Decode final hidden state into U, V
        h_final = self.final_mlp(h_t)
        U_flat, V_flat = torch.split(h_final, self.num_vars * self.rank, dim=1)
        U = U_flat.view(B, self.num_vars, self.rank)
        V = V_flat.view(B, self.num_vars, self.rank)

        coeffs = torch.bmm(U, V.transpose(1, 2))  # (B, p, p)
        preds = coeffs  # optionally you can apply some post-processing for preds

        return preds, coeffs



    

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Lightweight Temporal Encoder (efficient) ----------
class TemporalDilatedEncoderFast(nn.Module):
    def __init__(self, num_vars, hidden_dim=128, dilations=(1,2,4,8)):
        super().__init__()
        self.proj = nn.Linear(num_vars, hidden_dim)
        # Use small stack of depthwise separable friendly convs (fast)
        self.layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in dilations])

    def forward(self, x):  # x: (B,O,P)
        h = self.proj(x)                # (B,O,H)
        h = h.transpose(1,2)            # (B,H,O)
        for conv, ln in zip(self.layers, self.norms):
            z = conv(h)                 # (B,H,O)
            z = F.gelu(z)
            # LayerNorm expects (B,O,H) so transpose
            z_t = z.transpose(1,2)      # (B,O,H)
            z_t = ln(z_t)
            z = z_t.transpose(1,2)     # (B,H,O)
            h = h + z                  # residual (same shapes)
        h = h.transpose(1,2)            # (B,O,H)
        return h

# ---------- Deterministic Graph Learner (fast, cached) ----------
class GraphLearnerDeterministic(nn.Module):
    def __init__(self, num_vars, hidden_dim=128):
        super().__init__()
        self.num_vars = num_vars
        self.src = nn.Parameter(torch.randn(num_vars, hidden_dim) * 0.01)
        self.dst = nn.Parameter(torch.randn(num_vars, hidden_dim) * 0.01)
        self.scorer = nn.Linear(hidden_dim, 1)
        # cache
        self._cached_A = None
        self._cached_device = None
        self._cached_version = 0  # bump when force-recompute

    def _compute_logits(self):
        P = self.num_vars
        src = self.src.unsqueeze(1).expand(P, P, -1)   # (P,P,H)
        dst = self.dst.unsqueeze(0).expand(P, P, -1)   # (P,P,H)
        h = torch.tanh(src + dst)                      # (P,P,H)
        logits = self.scorer(h).squeeze(-1)            # (P,P)
        mask_eye = torch.eye(P, device=logits.device).bool()
        logits = logits.masked_fill(mask_eye, -1e9)
        return logits

    def forward(self, force_recompute=False):
        # compute deterministic adjacency via sigmoid(logits)
        device = self.src.device
        if (self._cached_A is None) or force_recompute or (self._cached_device != device):
            logits = self._compute_logits().to(device)
            A = torch.sigmoid(logits)   # (P,P) deterministic
            self._cached_A = A
            self._cached_device = device
            self._cached_version += 1
        else:
            A = self._cached_A
        # l0 proxy (mean) can be used if you want a reg term externally
        l0 = torch.mean(A)
        return A, l0

# ---------- Fast single-hop Message Passing (vectorized) ----------
class GraphMessagePassingFast(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.msg_out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_nodes, A):
        # h_nodes: (B,O,P,H)
        B, O, P, H = h_nodes.shape
        # flatten (B*O,P,H) for single matmul
        h_flat = h_nodes.view(B*O, P, H)                 # (B*O,P,H)
        # neighbor aggregation via A^T (fast single matmul)
        # new_nodes[bop] = A^T @ h_flat[bop]  -> (B*O, P, H)
        new_nodes = torch.matmul(A.transpose(0,1), h_flat)  # (B*O,P,H)
        new_nodes = self.msg_out(new_nodes)               # (B*O,P,H)
        out = (h_flat + new_nodes).view(B, O, P, H)        # residual
        return self.norm(out)

# ---------- Lightweight "MoE-lite" -> small residual MLP ----------
class MoELite(nn.Module):
    def __init__(self, hidden_dim=128, n_hidden=1):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):  # (B,O,P,H)
        out = self.net(h)
        return self.norm(h + out)

# ---------- Hybrid Coefficient Decoder (same idea, vectorized) ----------
class HybridCoeffDecoderFast(nn.Module):
    def __init__(self, num_vars, hidden_dim=128, rank=32):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.u_head = nn.Linear(hidden_dim, num_vars * rank)
        self.v_head = nn.Linear(hidden_dim, num_vars * rank)
        self.diag_head = nn.Linear(hidden_dim, num_vars)
        # remove huge sparse_head for speed (optional); keep lowrank+diag only
        # if you want sparse residual keep a small head, but it's heavier.
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h):  # h: (B,O,H)
        B, O, H = h.shape
        P = self.num_vars
        U = self.u_head(h).view(B, O, P, self.rank)    # (B,O,P,r)
        V = self.v_head(h).view(B, O, P, self.rank)    # (B,O,P,r)
        # lowrank via einsum
        lowrank = torch.einsum("boir,bojr->boij", U, V)   # (B,O,P,P)
        diag = torch.diag_embed(self.diag_head(h).view(B, O, P))  # (B,O,P,P)
        return lowrank + diag

# ---------- Fast Temporal Causal Model (vectorized and cached A) ----------
class TemporalCausalMoE(nn.Module):
    def __init__(self, num_vars, order, rank=32, hidden_dim=128,
                 dilations=(1,2,4,8), heads=8, n_experts=4, pe_scale=0.1,
                 l0_lambda=1e-4, fast_mode=True, use_fp16=False):
        """
        Same external signature as before but optimized.
        Set fast_mode=True to enable the fast path (default).
        """
        super().__init__()
        self.num_vars = num_vars
        self.order = order
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.fast_mode = fast_mode
        self.use_fp16 = use_fp16

        # lightweight temporal encoder
        self.temporal = TemporalDilatedEncoderFast(num_vars, hidden_dim, dilations)

        # positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, order, hidden_dim) * pe_scale)

        # deterministic graph learner (cached)
        self.graph = GraphLearnerDeterministic(num_vars, hidden_dim)

        # message passing (fast single matmul)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gmp = GraphMessagePassingFast(hidden_dim)

        # light MoE replacement
        self.moe = MoELite(hidden_dim, n_hidden=1)

        # coefficient decoder (lowrank + diag)
        self.coeff_decoder = HybridCoeffDecoderFast(num_vars, hidden_dim, rank)

    def forward(self, inputs, force_recompute_A=False):  # inputs: (B,O,P)
        import contextlib

        # optionally run in half precision
        if self.use_fp16 and inputs.is_cuda:
            dtype_switch = torch.cuda.amp.autocast(enabled=True)
        else:
            dtype_switch = contextlib.nullcontext()

        with dtype_switch:
            B, O, P = inputs.shape

            # 1) temporal features (vectorized)
            h_lag = self.temporal(inputs)                          # (B,O,H)
            # add pos enc (broadcast)
            pos = self.pos_enc[:, :O, :].to(h_lag.dtype).to(h_lag.device)
            h_lag = h_lag + pos

            # 2) broadcast to nodes (clone to avoid view inplace issues)
            h_nodes = self.node_proj(h_lag).unsqueeze(2).expand(B, O, P, self.hidden_dim).clone()  # (B,O,P,H)

            # 3) deterministic adjacency (cached) + message passing
            A, l0 = self.graph(force_recompute_A)   # (P,P), scalar proxy
            h_nodes = self.gmp(h_nodes, A)          # (B,O,P,H)

            # 4) light MoE (small MLP)
            h_nodes = self.moe(h_nodes)             # (B,O,P,H)

            # 5) aggregate nodes per lag
            h_agg = h_nodes.mean(dim=2)             # (B,O,H)

            # 6) decode coefficients vectorized
            coeffs_seq = self.coeff_decoder(h_agg)  # (B,O,P,P)

            # 7) vectorized prediction: einsum over lag and input var -> (B,P)
            preds = torch.einsum("boij,boj->bi", coeffs_seq, inputs)  # (B,P)

        return preds, coeffs_seq



