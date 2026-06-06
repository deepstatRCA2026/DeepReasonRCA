import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.GraphAttention import FeatureAttentionLayer
from models.embed import PositionalEmbedding



class AttentionLayer1(nn.Module):
    def __init__(self, attention, d_model, guidance_num, n_heads=4, d_keys=None,
                 d_values=None):
        super(AttentionLayer1, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        guidance_num = guidance_num
        pool_size = int(guidance_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, queries, keys, values, frequency_global_x, attn_mask):
        B, L, D = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        h = int(L ** 0.5)
        w = int(L ** 0.5)
        guidance_tokens = self.pool(frequency_global_x.reshape(B, h, w, D).permute(0, 3, 1, 2)).reshape(B, D, -1).permute(0, 2, 1)  # [128, 49, 64]
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            guidance_tokens,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class AttentionLayer2(nn.Module):
    def __init__(self, attention, d_model, guidance_num, n_heads=4, d_keys=None,
                 d_values=None):
        super(AttentionLayer2, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        guidance_num = guidance_num
        pool_size = int(guidance_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, queries, keys, values, norm_frequency_x, attn_mask):
        B, L, D = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        h = int(L ** 0.5)
        w = int(L ** 0.5)
        guidance_tokens = self.pool(norm_frequency_x.reshape(B, h, w, D).permute(0, 3, 1, 2)).reshape(B, D, -1).permute(0, 2, 1)  # [128, 25, 64]
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            guidance_tokens,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn



class EncoderLayer(nn.Module):
    def __init__(self, temporal_attention, frequency_attention, d_model, dropout=0.1, seq_len=100, num_nodes=55):
        super(EncoderLayer, self).__init__()
        self.temporal_attention = temporal_attention
        self.frequency_attention = frequency_attention
        self.feature_gat = FeatureAttentionLayer(n_features=num_nodes, window_size=seq_len, dropout=dropout, alpha=0.2, embed_dim=d_model, use_gatv2=True)
        self.embedding = PositionalEmbedding(d_model = d_model)

        self.linear1 = nn.Linear(num_nodes, d_model)
        self.linear2 = nn.Linear(seq_len * 2, seq_len)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x, attn_mask=None):
        x = self.feature_gat(x)
        x = self.linear1(x)

        x += self.embedding(x)

        temporal_x = x

        frequency_x = x.permute(0, 2, 1)
        frequency_fft_x = torch.fft.fft(frequency_x, dim=-1, norm='forward')
        frequency_x = torch.cat((frequency_fft_x.real, frequency_fft_x.imag), -3)
        frequency_x = torch.reshape(frequency_x.permute(1, 2, 0), [frequency_fft_x.shape[-3], frequency_fft_x.shape[-2], -1])
        frequency_x = frequency_x.permute(0, 2, 1)
        frequency_x = self.linear2(frequency_x.permute(0, 2, 1)).permute(0, 2, 1)


        new_x, attn = self.temporal_attention(
            temporal_x, temporal_x, temporal_x, frequency_x,
            attn_mask=attn_mask
        )
        temporal_x = temporal_x + self.dropout1(new_x)
        temporal_recons = self.norm1(temporal_x)


        new_x, attn = self.frequency_attention(
            frequency_x, frequency_x, frequency_x, temporal_x,
            attn_mask=attn_mask
        )
        frequency_x = frequency_x + self.dropout2(new_x)
        frequency_recons = self.norm2(frequency_x)


        return temporal_recons, frequency_recons, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None, d_model=128, out_dim=55):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm1 = norm_layer
        self.norm2 = norm_layer
        self.fusinTB = STSTransformerBlock(d_model, d_model)
        self.projection = nn.Linear(2 * d_model, out_dim)


    def forward(self, x, attn_mask=None):
        # x [B, D, L]
        temporal_recons = torch.Tensor
        frequency_recons = torch.Tensor

        for attn_layer in self.attn_layers:
            temporal_recons, frequency_recons, attn = attn_layer(x, attn_mask=attn_mask)

        if self.norm1 is not None:
            temporal_recons = self.norm1(temporal_recons)

        if self.norm2 is not None:
            frequency_recons = self.norm2(frequency_recons)

        fuse_out = self.fusinTB(temporal_recons.permute(0, 2, 1).unsqueeze(-2),
                                frequency_recons.permute(0, 2, 1).unsqueeze(-2))
        fuse_out = fuse_out.squeeze().permute(0, 2, 1)
        fuse_out = self.projection(fuse_out)

        return fuse_out



class STSTransformerBlock(nn.Module):
    def __init__(self, emb_size1, emb_size2, drop_p=0.1, forward_expansion=2, forward_drop_p=0.1):
        super().__init__()
        self.emb_size = emb_size1
        self.att_drop1 = nn.Dropout(drop_p)
        self.projection1 = nn.Linear(emb_size1, emb_size1)
        self.projection2 = nn.Linear(emb_size1, emb_size1)
        self.drop1 = nn.Dropout(drop_p)
        self.drop2 = nn.Dropout(drop_p)

        self.layerNorm1 = nn.LayerNorm(emb_size1)
        self.layerNorm2 = nn.LayerNorm(emb_size2)

        self.queries1 = nn.Linear(emb_size1, emb_size1)
        self.values1 = nn.Linear(emb_size1, emb_size1)
        self.keys2 = nn.Linear(emb_size2, emb_size2)
        self.values2 = nn.Linear(emb_size2, emb_size2)

        self.layerNorm3 = nn.LayerNorm(emb_size1 + emb_size2)
        self.drop3 = nn.Dropout(drop_p)

        self.ffb = nn.Sequential(
            nn.LayerNorm(emb_size1 + emb_size2),
            FeedForwardBlock(emb_size1 + emb_size2, expansion=forward_expansion, drop_p=forward_drop_p),
            nn.Dropout(drop_p)
        )

    def forward(self, x1, x2):
        x1 = rearrange(x1, 'b e (h) (w) -> b (h w) e ')
        x2 = rearrange(x2, 'b e (h) (w) -> b (h w) e ')
        res1 = x1
        res2 = x2

        x1 = self.layerNorm1(x1)
        x2 = self.layerNorm2(x2)
        queries1 = self.queries1(x1)
        values1 = self.values1(x1)
        keys2 = self.keys2(x2)
        values2 = self.values2(x2)

        energy = torch.einsum('bqd, bkd -> bqk', keys2, queries1)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop1(att)

        out1 = torch.einsum('bal, blv -> bav ', att, values1)
        out1 = self.projection1(out1)
        x1 = self.drop1(out1)
        x1 += res1

        out2 = torch.einsum('bal, blv -> bav ', att, values2)
        out2 = self.projection2(out2)
        x2 = self.drop2(out2)
        x2 += res2

        x = torch.cat((x1, x2), dim=-1)

        res = x
        x = self.ffb(x)
        x = self.drop3(self.layerNorm3(x + res))

        x = rearrange(x, 'b t e -> b e 1 t')
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
