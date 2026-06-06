import torch
from torch import dropout, dropout, nn
#from dgl.nn.pytorch import GATv2Conv
#from dgl.nn import GlobalAttentionPooling
from torch_geometric.nn import GATv2Conv, GlobalAttention
import torch.nn.functional as F


class GraphModel(nn.Module):
    def __init__(self, in_dim, graph_hiddens=[64, 128], attn_head=4, activation=0.2, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()

        for i, hidden in enumerate(graph_hiddens):
            in_feats = in_dim if i == 0 else graph_hiddens[i-1]
            self.layers.append(
                GATv2Conv(
                    in_channels=in_feats,
                    out_channels=hidden,
                    heads=attn_head,
                    concat=False,          # key line
                    negative_slope=activation
                )
            )

        self.out_dim = graph_hiddens[-1]

    def forward(self, A, x):
        num_nodes = A.shape[0]
        B = x.shape[0] // num_nodes

        edge_index = A.nonzero(as_tuple=False).t()
        edge_index = torch.cat(
            [edge_index + b * num_nodes for b in range(B)],
            dim=1
        )

        out = x
        for layer in self.layers:
            out = layer(out, edge_index)

        return out.view(B, num_nodes, -1)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dilation=2, **kwargs):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(len(kernel_sizes)):
            dilation_size = dilation ** i
            kernel_size = kernel_sizes[i]
            padding = (kernel_size-1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding), 
                       nn.BatchNorm1d(out_channels), nn.ReLU(), Chomp1d(padding)]
            
        self.network = nn.Sequential(*layers)
        
        self.out_dim = num_channels[-1]
        #self.network.to(dev)
        
    
    def forward(self, x): #[batch_size, T, in_dim]
        x = x.permute(0, 2, 1).float() #[batch_size, in_dim, T]
        out = self.network(x) #[batch_size, out_dim, T]
        out = out.permute(0, 2, 1) #[batch_size, T, out_dim]
        return out

import math
class SelfAttention(nn.Module):
    def __init__(self, input_size, seq_len):
        """
        Args:
            input_size: int, hidden_size * num_directions
            seq_len: window_size
        """
        super(SelfAttention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x):
        # x: [batch_size, window_size, input_size]
        input_tensor = x.transpose(1, 0)  # w x b x h
        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

class TraceModel(nn.Module):
    def __init__(self, node_num, trace_hiddens=[20, 50], trace_kernel_sizes=[3, 3], self_attn=False, chunk_lenth=None, **kwargs):
        super(TraceModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.net = ConvNet(node_num, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, **kwargs)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_lenth is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_lenth)

    def forward(self, x: torch.tensor): #[bz, T, 1]
        hidden_states = self.net(x)
        if self.self_attn: 
            return self.attn_layer(hidden_states)
        return hidden_states[:,-1,:] #[bz, out_dim]

class MetricModel(nn.Module):
    def __init__(self, metric_num, metric_hiddens=[64, 128], metric_kernel_sizes=[3, 3], self_attn=False, chunk_lenth=None, **kwargs):
        super(MetricModel, self).__init__()
        self.metric_num = metric_num
        self.out_dim = metric_hiddens[-1]
        in_dim = metric_num

        assert len(metric_hiddens) == len(metric_kernel_sizes)
        self.net = ConvNet(num_inputs=in_dim, num_channels=metric_hiddens, kernel_sizes=metric_kernel_sizes)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_lenth is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_lenth)

    
    def forward(self, x): #[bz, T, metric_num]
        assert x.shape[-1] == self.metric_num
        hidden_states = self.net(x)
        if self.self_attn: 
            return self.attn_layer(hidden_states)
        return hidden_states[:,-1,:] #[bz, out_dim]

class LogModel(nn.Module):
    def __init__(self, event_num, out_dim):
        super(LogModel, self).__init__()
        self.embedder = nn.Linear(event_num, out_dim) 
    def forward(self, paras: torch.tensor): #[bz, event_num]
        """
        Input:
            paras: mu with length of event_num
        """
        return self.embedder(paras)


class MultiSourceEncoder(nn.Module):
    def __init__(
        self,
        event_num,
        metric_num,
        node_num,
        log_dim=64,
        fuse_dim=64,
        alpha=0.5,
        **kwargs
    ):
        super().__init__()

        self.node_num = node_num
        self.alpha = alpha

        # --------------------------------------------------
        # modality encoders
        # --------------------------------------------------
        self.trace_model = TraceModel(node_num, **kwargs)
        self.log_model = LogModel(event_num, log_dim)
        self.metric_model = MetricModel(metric_num, **kwargs)

        trace_dim = self.trace_model.out_dim
        metric_dim = self.metric_model.out_dim

        # --------------------------------------------------
        # project every modality to common size
        # --------------------------------------------------
        self.common_dim = 64

        self.trace_proj = nn.Linear(trace_dim, self.common_dim)
        self.log_proj = nn.Linear(log_dim, self.common_dim)
        self.metric_proj = nn.Linear(metric_dim, self.common_dim)

        # --------------------------------------------------
        # fusion
        # --------------------------------------------------
        fuse_in = self.common_dim * 3

        if fuse_dim % 2 != 0:
            fuse_dim += 1

        self.fuse = nn.Linear(fuse_in, fuse_dim)
        self.activate = nn.GLU()

        self.feat_in_dim = fuse_dim // 2

        # --------------------------------------------------
        # graph encoder
        # --------------------------------------------------
        self.status_model = GraphModel(
            in_dim=self.feat_in_dim,
            **kwargs
        )

        self.feat_out_dim = self.status_model.out_dim

    def forward(self, A, data_node, data_log, data_edge):

        B = None
        N = self.node_num
        device = A.device

        # --------------------------------------------------
        # metric branch
        # --------------------------------------------------
        metric_embedding = None

        #if ndarray convert to tensor
        #TODO: ideally this should be handled outside the model, but we can keep it here for flexibility
        if isinstance(data_node, np.ndarray):
            data_node = torch.from_numpy(data_node).to(device)
        if isinstance(data_log, np.ndarray):
            data_log = torch.from_numpy(data_log).to(device)
        if isinstance(data_edge, np.ndarray):
            data_edge = torch.from_numpy(data_edge).to(device)

        if data_node is not None:
            B, T, N, D = data_node.shape

            metric_x = (
                data_node
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, D)
            )

            metric_embedding = self.metric_model(metric_x)
            metric_embedding = self.metric_proj(metric_embedding)

        # --------------------------------------------------
        # log branch
        # --------------------------------------------------
        log_embedding = None

        if data_log is not None:
            B, T, N, D = data_log.shape

            log_x = data_log.mean(dim=1)
            log_x = log_x.reshape(B * N, D)

            log_embedding = self.log_model(log_x)
            log_embedding = self.log_proj(log_embedding)

        # --------------------------------------------------
        # trace branch
        # --------------------------------------------------
        trace_embedding = None

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

            trace_x = (
                trace_x
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, D)
            )

            trace_embedding = self.trace_model(trace_x)
            trace_embedding = self.trace_proj(trace_embedding)

        # --------------------------------------------------
        # determine batch size
        # --------------------------------------------------
        if B is None:
            raise ValueError("No modality provided")

        # --------------------------------------------------
        # replace missing modalities with zeros
        # --------------------------------------------------
        shape = (B * N, self.common_dim)

        if metric_embedding is None:
            metric_embedding = torch.zeros(
                shape,
                device=device
            )

        if log_embedding is None:
            log_embedding = torch.zeros(
                shape,
                device=device
            )

        if trace_embedding is None:
            trace_embedding = torch.zeros(
                shape,
                device=device
            )

        # --------------------------------------------------
        # fusion
        # --------------------------------------------------
        feature = torch.cat(
            [
                trace_embedding,
                log_embedding,
                metric_embedding
            ],
            dim=-1
        )

        feature = self.activate(self.fuse(feature))

        embeddings = self.status_model(A, feature)

        return embeddings

class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, linear_sizes):
        super(FullyConnected, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i-1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor): #[batch_size, in_dim]
        return self.net(x)

import numpy as np




class MainModel(nn.Module):
    def __init__(self, event_num, metric_num, node_num,graph=None, debug=False, **kwargs):
        super(MainModel, self).__init__()

        self.node_num = node_num
        self.graph = graph

        # Encoder stays exactly as-is as the original Eadro model
        self.encoder = MultiSourceEncoder(
            event_num, metric_num, node_num,
            debug=debug, **kwargs
        )

        # Minimal reconstruction head (no extra structure), to unify loss calculation across models
        D = self.encoder.feat_out_dim
        F = event_num + metric_num + node_num
        self.reconstructor = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.Linear(D, F)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, data):#data_node, data_log, data_edge):
        data_node = data.get("metric", None)
        data_log = data.get("logts", None)
        data_edge = data.get("traces", None)
        
        A = self.graph.to(self.device) #TODO # ensure graph is on the same device as data

        B, T, N, _ = data_node.shape
        D = self.encoder.feat_out_dim
        embeddings = self.encoder(A, data_node, data_log, data_edge)                 # [B, D]
        recon = self.reconstructor(embeddings.view(B * N, D))
        recon = F.softplus(recon)
        recon = recon.view(B, N, -1)

        return recon  # [B, D]
        
