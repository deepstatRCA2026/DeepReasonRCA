import torch
import torch.nn as nn
import torch.nn.functional as F

class TexFilter(nn.Module):
    def __init__(self, embed_size, scale=0.02, sparsity_threshold=0.01,
                 use_gelu=False, use_swish=False, use_skip=False,
                 use_layernorm=False, hard_threshold=False,
                 use_window=False):
        super().__init__()
        self.embed_size = embed_size
        self.scale = scale
        self.sparsity_threshold = sparsity_threshold

        # Ablation flags
        self.use_gelu = use_gelu
        self.use_swish = use_swish
        self.use_skip = use_skip
        self.use_layernorm = use_layernorm
        self.hard_threshold = hard_threshold
        self.use_window = use_window

        self.w = nn.Parameter(self.scale * torch.randn(2, embed_size))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(embed_size))

        if self.use_layernorm:
            self.norm_real = nn.LayerNorm(embed_size, elementwise_affine=False)
            self.norm_imag = nn.LayerNorm(embed_size, elementwise_affine=False)
    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):  # x: [B, F, C] complex
        if self.use_window:
            window = torch.hann_window(x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)
            x = x * window  # Apply Hanning window

        x_real = x.real
        x_imag = x.imag

        # First layer
        o1_real = torch.einsum('bfc,c->bfc', x_real, self.w[0]) - torch.einsum('bfc,c->bfc', x_imag, self.w[1]) + self.rb1
        o1_imag = torch.einsum('bfc,c->bfc', x_imag, self.w[0]) + torch.einsum('bfc,c->bfc', x_real, self.w[1]) + self.ib1

        # Activation
        if self.use_gelu:
            o1_real = F.gelu(o1_real)
            o1_imag = F.gelu(o1_imag)
        elif self.use_swish:
            o1_real = self.swish(o1_real)
            o1_imag = self.swish(o1_imag)
        else:
            o1_real = F.relu(o1_real)
            o1_imag = F.relu(o1_imag)

        if self.use_skip:
            o1_real = o1_real + x_real
            o1_imag = o1_imag + x_imag

        # Second layer
        o2_real = torch.einsum('bfc,c->bfc', o1_real, self.w1[0]) - torch.einsum('bfc,c->bfc', o1_imag, self.w1[1]) + self.rb2
        o2_imag = torch.einsum('bfc,c->bfc', o1_imag, self.w1[0]) + torch.einsum('bfc,c->bfc', o1_real, self.w1[1]) + self.ib2

        # Hard or soft threshold
        if self.hard_threshold:
            o2_real = torch.where(o2_real.abs() < self.sparsity_threshold, 0.0, o2_real)
            o2_imag = torch.where(o2_imag.abs() < self.sparsity_threshold, 0.0, o2_imag)
        else:
            y = torch.stack([o2_real, o2_imag], dim=-1)
            y = F.softshrink(y, lambd=self.sparsity_threshold)
            o2_real, o2_imag = y.unbind(dim=-1)

        if self.use_layernorm:
            o2_real = self.norm_real(o2_real)
            o2_imag = self.norm_imag(o2_imag)

        y = torch.complex(o2_real, o2_imag)

        return y
