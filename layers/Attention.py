import torch
import torch.nn as nn

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


"""
Temporal Attention
"""
class TemporalAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, T=1, activation='softmax',
                 output_attention=False, guidance_num=25):
        super(TemporalAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.activation = activation
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.T = T

        self.guidance_num = guidance_num
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values, guidance_tokens, attn_mask):
        B, L, H, E = queries.shape
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        scale = H ** -0.5

        guidance_tokens = guidance_tokens.reshape(B, self.guidance_num, H, E).permute(0, 2, 1, 3)
        guidance_attn = self.softmax((guidance_tokens * scale) @ keys.transpose(-2, -1))
        guidance_v = guidance_attn @ values

        q_attn = self.softmax((queries * scale) @ guidance_tokens.transpose(-2, -1))
        x = q_attn @ guidance_v
        x = x.permute(0, 2, 1, 3)

        return (x.contiguous(), None)

"""
Fourier Attention
"""
class FourierAttention(nn.Module):
    def __init__(self, T=1, activation='softmax', output_attention=False, guidance_num=25):
        super(FourierAttention, self).__init__()
        self.activation = activation
        self.output_attention = output_attention
        self.T = T

        self.guidance_num = guidance_num
        self.softmax = nn.Softmax(dim=-1)



    def forward(self, queries, keys, values, guidance_tokens, mask):
        B, L, H, E = queries.shape
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        scale = H ** -0.5

        guidance_tokens = guidance_tokens.reshape(B, self.guidance_num, H, E).permute(0, 2, 1, 3)
        guidance_attn = self.softmax((guidance_tokens * scale) @ keys.transpose(-2, -1))
        guidance_v = guidance_attn @ values

        q_attn = self.softmax((queries * scale) @ guidance_tokens.transpose(-2, -1))
        x = q_attn @ guidance_v
        x = x.permute(0, 2, 1, 3)

        return (x.contiguous(), None)

