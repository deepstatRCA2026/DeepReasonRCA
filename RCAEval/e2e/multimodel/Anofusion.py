import torch as t
import numpy as np
import sys
sys.path.append('./')
from RCAEval.e2e.multimodel.inner_models.GTblock import GTN
from RCAEval.e2e.multimodel.inner_models.GATGRU import *
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class permute(nn.Module):
    def __init__(self):
        super(permute, self).__init__()
    def forward(self, x):
        return x.permute(0, 2, 1)


class AnoFusionWrapper(nn.Module):
    def __init__(
        self,
        num_services,
        window_size,
        metric_dim,
        log_dim,
        trace_dim,
        graph,
        hidden_dim=64,
        out_dim=32
    ):
        super().__init__()

        self.num_services = num_services
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.out_dim = metric_dim + log_dim + trace_dim

        # -----------------------------
        # modality projections
        # -----------------------------
        self.metric_proj = nn.Linear(metric_dim, hidden_dim)
        self.log_proj    = nn.Linear(log_dim, hidden_dim)
        self.trace_proj  = nn.Linear(trace_dim, hidden_dim)
        self.trace_dim = trace_dim
        # -----------------------------
        # fusion
        # -----------------------------
        self.fuse = nn.Linear(hidden_dim * 3, hidden_dim)

        # -----------------------------
        # backbone
        # -----------------------------
        self.anofusion = Net(
            node_num=num_services,
            edge_types=1,
            window_samples_num=window_size,
            num_vars=metric_dim + log_dim + trace_dim,
            dropout=0.2
        )

        # 🔥 FIX: infer output dim lazily (no assumption)
        self.post_proj = None
        self._initialized = False
        self.graph = graph
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_post_proj(self, sample_out):
        """
        Dynamically infer Net output dim
        """
        in_dim = sample_out.shape[-1]
        self.post_proj = nn.Linear(in_dim, self.out_dim).to(sample_out.device)
        self._initialized = True

    def forward(self, data):
        data_node = data["metric"]
        data_log  = data["logts"]
        data_edge = data["traces"]
        if isinstance(data_node, np.ndarray):
            data_node = torch.from_numpy(data_node).to(self.device)
        if isinstance(data_log, np.ndarray):
            data_log = torch.from_numpy(data_log).to(self.device)
        if isinstance(data_edge, np.ndarray):
            data_edge = torch.from_numpy(data_edge).to(self.device)
        if data_edge is not None:

            if data_edge.dim() == 5:
                B, T, N, _, D = data_edge.shape

                trace_x = data_edge.mean(dim=3)

            elif data_edge.dim() == 4:
                B, T, N, D = data_edge.shape
                trace_x = data_edge

            else:
                raise ValueError(
                    f"Unexpected trace shape: {data_edge.shape}"
                )
        else:
            trace_x = torch.zeros_like(data_node)

        graph     = self.graph.to(data_node.device)

        return self._forward(graph, data_node, data_log, trace_x)

    def _forward(self, graph, data_node, data_log, data_edge):

        B, T, N, _ = data_node.shape

        # -----------------------------
        # modality encoding
        # -----------------------------
        m = self.metric_proj(data_node)
        l = self.log_proj(data_log)
        if self.trace_dim > 0:
            t = self.trace_proj(data_edge)
        else:
            t = torch.zeros_like(m)

        x = torch.cat([m, l, t], dim=-1)
        x = self.fuse(x)

        # -----------------------------
        # temporal window
        # -----------------------------
        θ = min(self.window_size, T)
        x = x[:, -θ:]

        # -----------------------------
        # graph batching
        # -----------------------------
        A = graph.unsqueeze(0).unsqueeze(0).expand(B, θ, -1, -1, -1)

        x = x.reshape(B * θ, N, self.hidden_dim)
        A = A.reshape(B * θ, N, N, -1)

        # -----------------------------
        # backbone forward
        # -----------------------------
        out = self.anofusion(x, A, device=x.device)

        # -----------------------------
        # lazy projection init
        # -----------------------------
        if not self._initialized:
            self._build_post_proj(out)

        out = self.post_proj(out)
        out = F.relu(out)


        # reshape back to (B, θ, N, out_dim)
        out = out.view(B, θ, N, self.out_dim)
        # then mean pool over time dimension
        out = out.mean(dim=1)  # (B, N, out_dim)
        return out

class Net(nn.Module):
    def __init__(self, node_num, edge_types, window_samples_num,num_vars, dropout):
        super(Net, self).__init__()
        self.edge_types = edge_types
        self.num_channels = edge_types
        self.node_num = node_num
        self.window_samples_num = window_samples_num
        self.dropout = dropout
        self.GTN = GTN(edge_types=self.edge_types, num_channels=self.num_channels, num_layers=5, norm=False)
        self.GAT_GRU = GAT_GRU(self.window_samples_num, self.node_num, self.num_channels, num_vars)
        self.flatten = nn.Flatten()
        self.linT = nn.Linear(self.window_samples_num, self.window_samples_num // 2)
    
        self.all = self.window_samples_num * self.node_num
        self.Dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(self.all, 64)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(64, 2)
        self._final_softmax = nn.Softmax(dim=1)
      
        
    def forward(self, X, A, device):
        X = self.Dropout(X)
        A = A.view((-1, self.node_num, self.node_num, self.edge_types))
        X = X.view((-1, self.node_num, X.shape[-1]))
        # GTN
        #device = X.device
        A = self.GTN(A, device)
        # GAT and GRU
        out_T = self.GAT_GRU(X, A, device)
        return out_T