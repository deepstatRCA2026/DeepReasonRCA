import torch
import torch.nn as nn

class Model(nn.Module):
    """Mamba: Selective State Space (O(K) complexity)."""
    def __init__(self, configs):
        super().__init__()
        from mamba_ssm import Mamba

        input_dim = configs.enc_in  
        hidden_dim = configs.enc_in
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim, d_state=12, d_conv=4, expand=2)

    def forward(self, x):
        x = self.input_proj(x)
        return self.mamba(x)