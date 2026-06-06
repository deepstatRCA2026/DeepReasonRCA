import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class OrthTransform(nn.Module):
    def __init__(self, dataset_obj, device, eps=1e-3):
        super().__init__()

        self.device = device
        self.eps = eps

        print("Computing orthogonal whitening transform...")

        # ---- flatten full dataset ----
        x = dataset_obj.data_dict["x_n_list"]  # (S, W, V)
        x = x.reshape(-1, x.shape[-1])         # (S*W, V)

        # ---- covariance ----
        cov = np.cov(x, rowvar=False)

        # numerical stabilization
        cov = cov + eps * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])

        # ---- SVD whitening (stable) ----
        U, S, _ = np.linalg.svd(cov)
        inv_sqrt = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T

        self.register_buffer(
            "Q",
            torch.tensor(inv_sqrt, dtype=torch.float32, device=device)
        )

    def forward(self, x):
        return x @ self.Q

    def inverse(self, x):
        return x @ self.Q.T
    
    
class TemporalBlock(nn.Module):
    def __init__(self, seq_len, rank):
        super().__init__()

        self.enc = nn.Linear(seq_len, rank, bias=False)
        self.dec = nn.Linear(rank, seq_len, bias=False)

        self.mix = nn.Sequential(
            nn.Linear(rank, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, rank, bias=False),
        )

        self.norm = nn.LayerNorm(rank)

    def forward(self, x):
        """
        x: [B, D, T]
        """

        z = self.enc(x)            # [B,D,R]

        h = self.mix(z)
        z = self.norm(z + h)

        return self.dec(z)        # [B,D,T]

class Model(nn.Module):
    def __init__(self, options):
        super().__init__()

        self.seq_len = options.seq_len
        self.pred_len = options.seq_len

        self.enc_in = options.enc_in
        self.d_model = options.d_model

        # --------------------------------------------------
        # New low-rank temporal latent
        # --------------------------------------------------

        self.rank = getattr(options, "rank", 32)

        # --------------------------------------------------
        # Sensor embedding
        # --------------------------------------------------

        self.sensor_in = nn.Linear(
            self.enc_in,
            self.d_model,
            bias=False,
        )

        self.sensor_out = nn.Linear(
            self.d_model,
            self.enc_in,
            bias=False,
        )

        # --------------------------------------------------
        # Temporal compression
        # --------------------------------------------------

        self.time_enc = nn.Linear(
            self.seq_len,
            self.rank,
            bias=False,
        )

        self.time_dec = nn.Linear(
            self.rank,
            self.pred_len,
            bias=False,
        )

        # --------------------------------------------------
        # Temporal mixer
        # --------------------------------------------------

        self.mix = nn.Sequential(
            nn.Linear(self.rank, self.rank, bias=False),
            nn.GELU(),
            nn.Linear(self.rank, self.rank, bias=False),
        )

        self.norm = nn.LayerNorm(self.rank)

        # --------------------------------------------------
        # Whitening
        # --------------------------------------------------

        self.orth = getattr(options, "orth_transformer")

    def forward(self, x):
        """
        x : [B,T,C]
        """

        residual = x

        # --------------------------------------------------
        # Whitening
        # --------------------------------------------------

        x = self.orth(x)

        # --------------------------------------------------
        # Sensor embedding
        # --------------------------------------------------

        x = self.sensor_in(x)  # [B,T,D]

        # --------------------------------------------------
        # Time processing
        # --------------------------------------------------

        xt = x.transpose(1, 2)  # [B,D,T]

        z = self.time_enc(xt)   # [B,D,R]

        h = self.mix(z)

        z = self.norm(z + h)

        xt = self.time_dec(z)   # [B,D,T]

        # --------------------------------------------------
        # Back to [B,T,D]
        # --------------------------------------------------

        x = xt.transpose(1, 2)

        # --------------------------------------------------
        # Sensor reconstruction
        # --------------------------------------------------

        x = self.sensor_out(x)

        # --------------------------------------------------
        # Global residual
        # --------------------------------------------------

        return x + residual