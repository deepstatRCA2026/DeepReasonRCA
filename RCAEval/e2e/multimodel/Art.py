import dgl
import sys
#dgl.heterograph.DGLHeteroGraph = dgl.DGLGraph
#if hasattr(dgl, 'DGLGraph'):
#    sys.modules['dgl.heterograph'] = dgl.heterograph
#    setattr(dgl.heterograph, 'DGLHeteroGraph', dgl.DGLGraph)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import dgl


class ARTWrapper(nn.Module):
    def __init__(
        self,
        adj,
        raw_metric_dim,
        raw_log_dim,
        raw_trace_dim,
        hidden_dim=64,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # -----------------------------
        # projections (raw -> shared)
        # -----------------------------
        self.metric_proj = nn.Linear(raw_metric_dim, hidden_dim)
        self.log_proj    = nn.Linear(raw_log_dim, hidden_dim)
        self.trace_proj  = nn.Linear(raw_trace_dim, hidden_dim)

        # -----------------------------
        # build static graph
        # -----------------------------
        if torch.is_tensor(adj):
            adj_np = adj.detach().cpu().numpy()
        else:
            adj_np = adj

        src, dst = np.nonzero(adj_np)
        num_nodes = adj_np.shape[0]

        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g = dgl.add_self_loop(g)

        self.register_buffer("adj_matrix", torch.tensor(adj_np, dtype=torch.float32))
        self.g = g

        # -----------------------------
        # fusion + graph encoder
        # -----------------------------
        gnn_in_dim = hidden_dim * 3

        self.gnn = AutoRegressor(
            tf_in_dim=num_nodes,
            num_heads=2,
            gnn_in_dim=gnn_in_dim,
            gnn_hidden_dim=64,
            gnn_out_dim=32,
            gru_hidden_dim=32,
            dropout=0.3,
            tf_layers=1,
            gnn_layers=2,
            gru_layers=1
        )

        self.out = nn.Linear(gnn_in_dim, raw_metric_dim + raw_log_dim + raw_trace_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self,data):# metrics, logs, traces):
        """
        metrics: [B,T,N,Fm]
        logs:    [B,T,N,Fl]
        traces:  [B,T,N,N,Ft]
        """
        #data_node = data.get("metric", None)
        #data_log = data.get("logts", None)
        #data_edge = data.get("traces", None)
        metrics = data.get("metric", None)
        logs = data.get("logts", None)
        traces = data.get("traces", None)


        #device = metrics.device
        B, T, N, _ = metrics.shape

        # -----------------------------
        # project modalities
        # -----------------------------
        if isinstance(metrics, np.ndarray):
            metrics = torch.from_numpy(metrics).to(self.device)
        if isinstance(logs, np.ndarray):
            logs = torch.from_numpy(logs).to(self.device)
        if isinstance(traces, np.ndarray):
            traces = torch.from_numpy(traces).to(self.device)
        if traces is not None:
            if traces.dim() == 4:
                #expand to 5D
                B, T, N, C = traces.shape
                traces = traces.unsqueeze(-2).expand(B, T, N, N, C)

        m = self.metric_proj(metrics)              # [B,T,N,H]
        l = self.log_proj(logs)                    # [B,T,N,H]
        if traces is not None:
            t = self.trace_proj(traces).mean(dim=3)    # [B,T,N,H]
        else:
            t = torch.zeros_like(m)                    # [B,T,N,H]
        # -----------------------------
        # fuse
        # -----------------------------
        x = torch.cat([m, l, t], dim=-1)  # [B,T,N,3H]

        # reshape for GNN: [B*N, T, 3H]
        #x = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)

        # -----------------------------
        # graph encoding
        # -----------------------------
        h = self.gnn(self.g, x)  # [B*N, T, H]

        # pool time dimension
        #h = h.mean(dim=1)        # [B*N, H]

        # -----------------------------
        # reconstruction
        # -----------------------------
        recon = self.out(h)      # [B*N, F]
        recon = F.relu(recon)

        return recon#recon.view(B, N, -1)


# -------------------- Neural Network Architecture -------------------------
# Transformer - Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=1,
            dim_feedforward=in_dim*2,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

    def forward(self, features):
        # (batch_size, sequence_length, hidden_size)
        # print(x.shape)
        h = F.leaky_relu(self.transformer_encoder(features))
        return h

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm):
        super(GraphSAGEEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_dim = hidden_dim if num_layers > 1 else out_dim
        self.input_conv = dgl.nn.GraphConv(in_dim, hidden_dim, norm=norm)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(dgl.nn.GraphConv(hidden_dim, hidden_dim, norm=norm))
        if num_layers > 1:
            self.convs.append(dgl.nn.GraphConv(hidden_dim, out_dim, norm=norm))

    def forward(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
            h = self.dropout(h)
        return h
    
    def transform(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
        return h
class Extractor(nn.Module):
    def __init__(self, tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout=0, tf_layers=1, gnn_layers=2, gru_layers=1):
        super(Extractor, self).__init__()
        self.TFEncoder = TransformerEncoder(tf_in_dim, num_heads, tf_layers)
        self.GRUEncoder = nn.GRU(gnn_in_dim, gru_hidden_dim, gru_layers, bias=False, batch_first=True)
        self.GraphEncoder = GraphSAGEEncoder(gru_hidden_dim, gnn_hidden_dim, gnn_out_dim, gnn_layers, dropout, norm='none')
        
    def forward(self, g, features):
        bacth_size, series_len, instance_num, channel_dim = features.shape # 2,5,46,130
        h = features.permute(0,1,3,2)
        h = h.view(-1, channel_dim, instance_num)
        h = self.TFEncoder(h)
        h = h.permute(0,2,1).view(bacth_size, series_len, instance_num, channel_dim).permute(0,2,1,3).reshape(-1, series_len, channel_dim) # 92,5,130
        output, h_n = self.GRUEncoder(h)
        h = F.leaky_relu(h_n[-1]) # 92,32
        g_batches = dgl.batch([g] * bacth_size) #as the graph is static, same graph for each instance
        g_batches = g_batches.to(h.device)
        h = self.GraphEncoder(g_batches, h) # 92, 32
        h = h.view(bacth_size, instance_num, -1) # 2,46,32
        return h
    
class Regressor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Regressor, self).__init__()
        self.mlp = nn.Linear(in_dim, out_dim)
        
    def forward(self, features):
        h = F.leaky_relu(self.mlp(features))
        return h

class AutoRegressor(nn.Module):
    def __init__(self, tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout=0, tf_layers=1, gnn_layers=2, gru_layers=1):
        super(AutoRegressor, self).__init__()
        self.extractor = Extractor(tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout, tf_layers, gnn_layers, gru_layers)
        self.regressor = Regressor(gru_hidden_dim, gnn_in_dim)
        
    def forward(self, g, features):
        z = self.extractor(g, features)
        h = self.regressor(z)
        #return z, h
        return h #reconstructed features
# end Neural Network Architecture -----------------------------------------

