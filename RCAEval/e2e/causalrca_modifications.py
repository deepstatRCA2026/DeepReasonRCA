
class MLPEncoder_new(nn.Module):
    """Optimized MLP encoder module."""
    def __init__(
        self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0.0, factor=True, tol=0.1
    ):
        super().__init__()
        adj_A = torch.from_numpy(adj_A).double()
        self.adj_A = nn.Parameter(adj_A)
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out, dtype=torch.double))
        self.fc1 = nn.Linear(n_xdims, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.dropout_prob = do_prob
        self.batch_size = batch_size

        self.z = nn.Parameter(torch.tensor(tol, dtype=torch.double))
        self.z_positive = nn.Parameter(torch.ones_like(adj_A))
        self.register_buffer("eye", torch.eye(adj_A.size(0), dtype=torch.double))  # cache eye

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()

    def forward(self, inputs):
        if torch.isnan(self.adj_A).any():
            print("nan error in adj_A\n")

        # fast adjacency amplification (no gradient needed here)
        with torch.no_grad():
            adj_A1 = torch.sinh(3.0 * self.adj_A)
            adj_Aforz = preprocess_adj_new(adj_A1)

        H1 = F.relu(self.fc1(inputs))
        x = self.fc2(H1)
        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa

        return x, logits, adj_A1, self.eye, self.z, self.z_positive, self.adj_A, self.Wa



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

# small constant used instead of -inf when masking
_NEG_INF = -1e9

class SpatialAttentionGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, alpha=0.2):
        super().__init__()
        # Use float32 to match default Linear dtype unless you deliberately want double
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # attention vectors
        self.a_src = nn.Parameter(torch.empty(out_dim, 1, dtype=torch.float32))
        self.a_dst = nn.Parameter(torch.empty(out_dim, 1, dtype=torch.float32))

        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x: [B, N, F] (float32)
        adj: [N, N] or [B, N, N]  (will be coerced to x.dtype and device)
        """
        # ensure adj on same device/dtype
        device = x.device
        dtype = x.dtype

        if adj.dtype != dtype:
            adj = adj.to(dtype)
        if adj.device != device:
            adj = adj.to(device)

        # project features
        x = x.to(torch.float32)
        
        h = self.W(x)  # [B, N, out_dim]
        B, N, out_dim = h.shape

        # compute e_src and e_dst without forming big 4D tensors
        # torch.matmul(h, a_src) -> [B, N, 1]
        e_src = torch.matmul(h, self.a_src)  # [B, N, 1]
        e_dst = torch.matmul(h, self.a_dst)  # [B, N, 1]

        # expand to [B, N, N] cheaply (these are small 3D tensors)
        e_src = e_src.expand(B, N, N)        # broadcasting
        e_dst = e_dst.transpose(1, 2).expand(B, N, N)

        e = self.leakyrelu(e_src + e_dst)    # [B, N, N]

        # ensure adj is batched
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        # Add self-loops to avoid zero-degree rows (guarantees at least one valid entry)
        eye = torch.eye(N, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
        adj_with_loops = (adj + eye).clamp(min=0)  # ensure non-negative

        # mask non-edges using a large negative number (not -inf)
        e = e.masked_fill(adj_with_loops == 0, _NEG_INF)

        # softmax (dim=-1). Use nan_to_num to be safe.
        attn = F.softmax(e, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        # dropout on attention
        attn = self.dropout(attn)

        # compute messages: batch matrix multiply
        h_prime = torch.bmm(attn, h)  # [B, N, out_dim]

        # extra safety: if any NaNs appear, print debug info (remove in prod)
        if torch.isnan(h_prime).any():
            print("Warning: NaN in h_prime; debug stats:",
                  "max e", torch.nanmax(e).item() if torch.isfinite(e).any() else float('nan'),
                  "min e", torch.nanmin(e).item() if torch.isfinite(e).any() else float('nan'),
                  "adj sum", adj_with_loops.sum().item())

        return F.elu(h_prime)


class MLPEncoder_graph(nn.Module):
    """Encoder with spatial attention GNN as first layer (stable/dtype-safe)."""
    def __init__(
        self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size,
        do_prob=0.0, factor=True, tol=0.1
    ):
        super().__init__()
        # convert adjacency to float32 to match Linear default dtype
        adj_A = torch.from_numpy(adj_A).float()
        self.adj_A = nn.Parameter(adj_A)  # [N, N]
        self.factor = factor

        # Wa shaped [n_out] float32
        self.Wa = nn.Parameter(torch.zeros(n_out, dtype=torch.float32))

        # GNN first layer
        self.gnn1 = SpatialAttentionGNNLayer(n_xdims, n_hid, dropout=do_prob)
        self.fc2 = nn.Linear(n_hid, n_out)

        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol, dtype=torch.float32))
        self.z_positive = nn.Parameter(torch.ones_like(self.adj_A))
        self.register_buffer("eye", torch.eye(self.adj_A.size(0), dtype=torch.float32))

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        """
        x: [B, N, F] input features (float32)
        Returns:
            x_proj: [B, N, n_out] projected node features
            logits: [B, N, n_out] adjacency-transformed output
            adj_A1: [N, N] processed adjacency
            eye: [N, N] identity
            z: scalar parameter
            z_positive: [N, N] positive matrix
            adj_A: [N, N] adjacency parameter
            Wa: [n_out] bias
        """
        # ensure float32 and device consistency
        x = x.float()
        device = x.device
        dtype = x.dtype

        adj = self.adj_A.float().to(device=device)
        Wa = self.Wa.float().to(device=device)

        # 1️⃣ Run GNN layer
        h = self.gnn1(x, adj)  # [B, N, n_hid]

        # 2️⃣ Project to output dimension
        x_proj = self.fc2(h)   # [B, N, n_out]

        # 3️⃣ Prepare adjacency transform
        with torch.no_grad():
            adj_clamped = torch.clamp(adj, -3.0, 3.0)
            adj_A1 = torch.clamp(torch.sinh(3.0 * adj_clamped), -1e3, 1e3)
            adj_Aforz = preprocess_adj_new(adj_A1).float().to(device=device)

        # 4️⃣ Expand adjacency for batch
        B, N, _ = x_proj.shape
        adj_Aforz_batched = adj_Aforz.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        # 5️⃣ Compute logits safely
        # Wa broadcast along last dimension
        Wa_exp = Wa.unsqueeze(0).unsqueeze(0)  # [1, 1, n_out]
        logits = torch.bmm(adj_Aforz_batched, x_proj + Wa_exp) - Wa_exp

        # 6️⃣ Optional safety check for NaNs
        if torch.isnan(logits).any():
            print("Warning: NaN detected in logits. adj_A1 stats:",
                "min", torch.nanmin(adj_A1).item(),
                "max", torch.nanmax(adj_A1).item())

        return x_proj, logits, adj_A1, self.eye.float().to(device), self.z.float().to(device), \
           self.z_positive.float().to(device), adj, Wa

class MLPDecoder_new(nn.Module):
    """Optimized MLP decoder module."""
    def __init__(
        self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size, n_hid, do_prob=0.0
    ):
        super().__init__()
        self.out_fc1 = nn.Linear(n_in_z, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_out)
        self.batch_size = batch_size
        self.data_variable_size = data_variable_size
        self.dropout_prob = do_prob
        self.register_buffer("eye", torch.eye(data_variable_size, dtype=torch.double))  # cache eye
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.out_fc1.weight)
        nn.init.xavier_normal_(self.out_fc2.weight)
        self.out_fc1.bias.data.zero_()
        self.out_fc2.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, origin_A, adj_A_tilt, Wa):
        # Precompute adjacency transform with no_grad for speed
        with torch.no_grad():
            adj_A_new1 = preprocess_adj_new1(origin_A)

        mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa
        H3 = F.relu(self.out_fc1(mat_z))
        out = self.out_fc2(H3)
        return mat_z, out, adj_A_tilt

