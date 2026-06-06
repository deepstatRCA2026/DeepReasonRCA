import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# 1. SFlex-style temporal encoder (modality-specific)
# ======================================================
class SFlexEncoder(nn.Module):
    def __init__(self, enc_in, d_model, seq_len):
        super().__init__()

        self.enc_in = enc_in
        self.d_model = d_model
        self.seq_len = seq_len

        self.sensor_in = nn.Linear(enc_in, d_model, bias=False)
        self.sensor_out = nn.Linear(d_model, d_model, bias=False)

        self.time_enc = nn.Linear(seq_len, d_model, bias=False)
        self.time_mix = nn.Parameter(torch.eye(d_model))

    def forward(self, x):
        """
        x: [B, T, F]
        """

        B, T, F = x.shape

        x = self.sensor_in(x)                  # [B,T,D]

        x_t = x.transpose(1, 2)                # [B,D,T]
        s = torch.tanh(self.time_enc(x_t))     # [B,D,D]

        s = s @ self.time_mix                  # [B,D,D]

        y = s @ x_t                            # [B,D,T]
        y = y.transpose(1, 2)                  # [B,T,D]

        return self.sensor_out(y)              # [B,T,D]


# ======================================================
# 2. Multimodal SFlexRCA Encoder
# ======================================================

class MultiSourceEncoder(nn.Module):
    def __init__(self, metric_dim, log_dim, trace_dim, seq_len, hidden=64):
        super().__init__()

        self.hidden = hidden

        # modality encoders
        self.metric_enc = SFlexEncoder(metric_dim, hidden, seq_len)
        self.log_enc    = SFlexEncoder(log_dim, hidden, seq_len)
        self.trace_enc  = SFlexEncoder(trace_dim, hidden, seq_len)

        # project all modalities to same size
        self.common_dim = hidden

        # concatenate metric + log + trace
        fuse_in = self.common_dim * 3

        # GLU halves dimensions, therefore output 2*hidden first
        self.fuse = nn.Linear(fuse_in, hidden * 2)
        self.act = nn.GLU(dim=-1)

        # final embedding size = hidden
        self.out_proj = nn.Linear(hidden, hidden)

        self.feat_out_dim = hidden

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _to_tensor(self, x):
        if x is None:
            return None

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        return x.float().to(self.device)

    def forward(self, metric=None, logts=None, traces=None):

        metric = self._to_tensor(metric)
        logts  = self._to_tensor(logts)
        traces = self._to_tensor(traces)

        B = None
        N = None

        metric_feat = None
        log_feat = None
        trace_feat = None

        # ---------------- metric ----------------
        if metric is not None:
            B, T, N, F = metric.shape

            x = (
                metric
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, F)
            )

            metric_feat = self.metric_enc(x).mean(dim=1)

        # ---------------- logs ----------------
        if logts is not None:
            B, T, N, F = logts.shape

            x = (
                logts
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, F)
            )

            log_feat = self.log_enc(x).mean(dim=1)

        # ---------------- traces ----------------
        if traces is not None:

            if traces.dim() == 5:
                B, T, N, _, F = traces.shape
                x = traces.mean(dim=3)

            elif traces.dim() == 4:
                B, T, N, F = traces.shape
                x = traces

            else:
                raise ValueError(
                    f"Unexpected trace shape {traces.shape}"
                )

            x = (
                x
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, F)
            )

            trace_feat = self.trace_enc(x).mean(dim=1)

        if B is None:
            raise ValueError("No modality provided")

        shape = (B * N, self.common_dim)

        if metric_feat is None:
            metric_feat = torch.zeros(shape, device=self.device)

        if log_feat is None:
            log_feat = torch.zeros(shape, device=self.device)

        if trace_feat is None:
            trace_feat = torch.zeros(shape, device=self.device)

        # ---------------- fusion ----------------
        x = torch.cat(
            [
                trace_feat,
                log_feat,
                metric_feat,
            ],
            dim=-1
        )

        x = self.fuse(x)      # [B*N, 128]
        x = self.act(x)       # [B*N, 64]
        x = self.out_proj(x)  # [B*N, 64]

        return x

# ======================================================
# 3. Main Model (Eadro-compatible)
# ======================================================
class MainModel(nn.Module):
    def __init__(self, event_num, metric_num, node_num, seq_len=12, hidden=64):
        super().__init__()
        # log, metric, trace dimensions
        self.node_num = node_num
        self.encoder = MultiSourceEncoder(
            metric_dim=metric_num,
            log_dim=event_num,
            trace_dim=node_num,
            seq_len=seq_len,
            hidden=hidden
        )

        F = event_num + metric_num + node_num

        self.reconstructor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, F)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        metric = data.get("metric", None)
        logts  = data.get("logts", None)
        traces = data.get("traces", None)

        device = self.device

        # (graph not used, but kept for compatibility)
        B, T, N, _ = metric.shape
        A = torch.eye(N, device=device)

        emb = self.encoder(metric, logts, traces)  # [B*N, H]

        recon = self.reconstructor(emb)
        recon = recon.view(B, N, -1)

        return recon
    



    