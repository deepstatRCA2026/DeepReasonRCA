import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from RCAEval.e2e.models.texfilter import TexFilter  # Assuming TexFilter is in the same directory
class Model(nn.Module):
    # Hybrid FITS: RIN + Learnable Frequency Filtering (TexFilter) + Interpolation
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList([
                nn.Linear((self.seq_len // 2 + 1), int((self.seq_len + self.pred_len) // 2 + 1)).to(torch.cfloat)
                for _ in range(self.channels)
            ])
        else:
            self.freq_upsampler = nn.Linear((self.seq_len // 2 + 1),
                                (self.seq_len // 2 + 1))
            self.length_ratio = 1.0

        # NEW: Learnable frequency filter
        #self.texfilter = TexFilter(embed_size=self.channels,
        #                            use_gelu=False,
        #                            use_skip=False,
        #                            use_layernorm=False,
        #                            hard_threshold=False,
        #                            use_window=False)
        self.texfilter =    TexFilter(
                        embed_size=self.channels,
                        use_gelu=True,             # or use_swish=True for smoother nonlinearity
                        use_skip=True,             # ✅ Preserve original signal paths
                        use_layernorm=True,        # ✅ Stabilize across frequency bins
                        hard_threshold=False,      # ❌ Avoid hard cutting off weak signals
                        use_window=False,          # ❌ Avoid muting boundary info
                        sparsity_threshold=0.0     # ✅ Retain all weak signal components
                    )
        """
        🔝 Most Impactful Modifications (Ranked)
        Rank	Feature	                Expected Impact	    Notes
        1️⃣	use_skip (Skip connection)	⭐⭐⭐⭐	      Strongly stabilizes learning and gradient flow; allows feature reuse; widely helpful in deep and shallow nets alike.
        2️⃣	use_layernorm	            ⭐⭐⭐⭐	      Helps convergence and generalization, especially with long or dynamic sequences. Particularly useful for frequency-domain data which may have diverse scales.
        3️⃣	use_gelu / use_swish	    ⭐⭐⭐	           Both provide smoother nonlinearities than ReLU, improving expressiveness without hurting gradient flow. Swish is theoretically slightly better, but more expensive.
        4️⃣	hard_threshold	            ⭐⭐	             Forces exact sparsity. Can be helpful if you're modeling truly sparse frequency signals (e.g., anomalies), but risks killing useful low-energy signals.
        5️⃣	use_window	                ⭐⭐	             Applies a Hanning window in frequency. Helps with spectral leakage, but may discard useful boundary info in short sequences. Often useful in clean forecasting, but might be redundant in learned models.
        """
    def forward(self, x):
        # ---------------------
        # 1. RevIN (manual)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        # ---------------------
        # 2. rFFT
        specx = torch.fft.rfft(x, dim=1)  # [B, F, C]

        # ---------------------
        # 3. Apply TexFilter (learnable frequency attention)
        specx = specx * self.texfilter(specx)

        # ---------------------
        # 4. Interpolate in frequency domain
        # specx is complex: [B, F, C]
        specx_real = specx.real
        specx_imag = specx.imag

        if self.individual:
            specxy_real = torch.zeros_like(specx_real)
            specxy_imag = torch.zeros_like(specx_imag)
            for i in range(self.channels):
                # Use normal Linear (float) on real and imag separately
                specxy_real[:, :, i] = self.freq_upsampler[i](specx_real[:, :, i].permute(0, 1)).permute(0, 1)
                specxy_imag[:, :, i] = self.freq_upsampler[i](specx_imag[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            specxy_real = self.freq_upsampler(specx_real.permute(0, 2, 1)).permute(0, 2, 1)
            specxy_imag = self.freq_upsampler(specx_imag.permute(0, 2, 1)).permute(0, 2, 1)

        specxy_ = torch.complex(specxy_real, specxy_imag)
        # ---------------------
        # 5. Pad if needed
        # Match shapes before computing loss
        if specxy_.size(1) < x.size(1):
            # pad one step at the end
            pad_len = x.size(1) - specxy_.size(1)
            specxy_ = F.pad(specxy_, (0, 0, 0, pad_len))  # pad along time dim
        elif specxy_.size(1) > x.size(1):
            specxy_ = specxy_[:, :x.size(1), :]
        # ---------------------
        # 6. Inverse FFT
        low_xy = torch.fft.irfft(specxy_, n=x.size(1), dim=1)
        low_xy = low_xy * self.length_ratio

        # ---------------------
        # 7. Reverse RIN
        xy = (low_xy * torch.sqrt(x_var)) + x_mean
        return xy#, low_xy * torch.sqrt(x_var)
