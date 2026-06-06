from layers.Myformer_EncDec import EncoderLayer, Encoder, AttentionLayer1, AttentionLayer2
import torch.nn as nn
import torch
from layers.Attention import FourierAttention, TemporalAttention


class TS_Model(nn.Module):
    """
    normal pattern learning
    """
    def __init__(self, seq_len, num_nodes, d_model):
        super(TS_Model, self).__init__()
        # Attention
        frequencyAttention = FourierAttention(guidance_num=25)

        temporalAttention = TemporalAttention(False, attention_dropout=0.1, guidance_num=25)

        self.num_nodes = num_nodes
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer1(temporalAttention, d_model=d_model, guidance_num=25),
                    AttentionLayer2(frequencyAttention, d_model=d_model, guidance_num=25),
                    d_model=d_model,
                    dropout=0.1,
                    seq_len=seq_len,
                    num_nodes=num_nodes
                )
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            d_model=d_model,
            out_dim=num_nodes
        )



    def forward_orinalg(self, x):
        output = self.encoder(x)
        output = output[:, -1, :]      # Take only the last time step
        return output, None


    def forward(self, x, batch_chunk_size: int = 32):
        """
        x: (B, T, P)
        returns: (B, P) using chunked processing
        """
        B, T, P = x.shape
        device = x.device
        preds_list = []

        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)
            x_chunk = x[start:end]             # (B_chunk, T, P)
            output_chunk = self.encoder(x_chunk)  # (B_chunk, T, P)
            preds_chunk = output_chunk[:, -1, :]  # Take last time step (B_chunk, P)
            preds_list.append(preds_chunk)

        preds = torch.cat(preds_list, dim=0)    # (B, P)
        return preds, None