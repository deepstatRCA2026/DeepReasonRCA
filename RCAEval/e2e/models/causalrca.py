import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

cuda = torch.cuda.is_available()

def preprocess_adj_new(adj):  # NOTE
    if cuda:
        adj_normalized = torch.eye(adj.shape[0]).double().cuda() - (adj.transpose(0, 1))
    else:
        adj_normalized = torch.eye(adj.shape[0]).double() - (adj.transpose(0, 1))
    return adj_normalized


def preprocess_adj_new1(adj):  # NOTE
    if cuda:
        adj_normalized = torch.inverse(
            torch.eye(adj.shape[0]).double().cuda() - adj.transpose(0, 1)
        )
    else:
        adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double() - adj.transpose(0, 1))
    return adj_normalized


class MLPEncoder(nn.Module):  # NOTE
    """MLP encoder module."""

    def __init__(
        self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0.0, factor=True, tol=0.1
    ):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias=True)#12
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        if torch.sum(self.adj_A != self.adj_A):
            print("nan error \n")

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.0 * self.adj_A)

        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj_new(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0]).double()
        H1 = F.relu((self.fc1(inputs)))
        x = self.fc2(H1)
        #logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa
        # expand adj_Aforz to [1, 44, 44] → [batch, 44, 44]
        adj_batch = adj_Aforz.unsqueeze(0).expand(x.size(0), -1, -1)  # [64, 44, 44]

        # transpose x to [batch, features, seq_len] for matmul
        x_t = x.transpose(1, 2)  # [64, 44, 12]
        x_t = x_t.double()  # convert input to Double
        # matmul: [batch, 44, 44] @ [batch, 44, 12] → [64, 44, 12]
        logits_t = torch.matmul(adj_batch, x_t + self.Wa.view(1, -1, 1))

        # transpose back to [batch, seq_len, features]
        logits = logits_t.transpose(1, 2) - self.Wa  # [64, 12, 44]

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa


class MLPDecoder(nn.Module):  # NOTE
    """MLP decoder module."""

    def __init__(
        self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size, n_hid, do_prob=0.0
    ):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias=True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #def forward(self, inputs, input_z, n_in_node, origin_A, adj_A_tilt, Wa):
    #    # adj_A_new1 = (I-A^T)^(-1)
    #    adj_A_new1 = preprocess_adj_new1(origin_A)
    #    mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa
#
    #    H3 = F.relu(self.out_fc1((mat_z)))
    #    out = self.out_fc2(H3)
#
    #    return mat_z, out, adj_A_tilt
    def forward(self, inputs, input_z, n_in_node, origin_A, adj_A_tilt, Wa):
        """
        inputs: [batch, seq_len, num_vars]
        returns: 
            enc_x: [batch, seq_len, n_out]
            logits: [batch, seq_len, n_out]  # causal adjacency applied
            origin_A: original adjacency
            adj_A_tilt: transformed adjacency
            z_gap, z_positive, myA, Wa: optional
        """
        # Ensure inputs are float32
        inputs = inputs.double()

        # preprocess adjacency for causal propagation
        adj_Aforz = preprocess_adj_new(origin_A)  # should be [num_vars, num_vars], float32

        # Broadcast Wa to match x shape
        Wa = Wa.view(1, 1, -1).double()  # [1,1,num_vars]

        # Forward through MLP
        H1 = F.relu(self.out_fc1(inputs))  # [batch, seq_len, n_hid]
        out = self.out_fc2(H1)               # [batch, seq_len, n_out]

        # Apply adjacency: batch matmul requires [batch, seq_len, n_out] × [n_out, n_out]? 
        # Actually we want adjacency applied to features (last dim)
        out = out.double()  # convert to double for matmul
        adj_Aforz = adj_Aforz.double()  # ensure adj is float32
        mat_z = torch.matmul(out + Wa, adj_Aforz.T) - Wa  # [batch, seq_len, n_out]

        return mat_z, out, adj_A_tilt


class Model(nn.Module):
    def __init__(
        self,
        configs
    ):
        super(Model, self).__init__()
        n_in = configs.data_variable_size
        n_xdims = configs.n_xdims
        n_hid = configs.n_hid
        n_out = int(configs.n_out)
        adj_A = configs.adj_A
        batch_size = configs.batch_size
        do_prob=0.0
        factor=True
        tol=0.1

        self.encoder = MLPEncoder(
            n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob, factor, tol
        )
        self.decoder = MLPDecoder(
            n_in,
            n_out,
            n_out,
            self.encoder,
            configs.data_variable_size,
            batch_size,
            n_hid,
            do_prob,
        )

    def forward(self, inputs):
        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = self.encoder(inputs)
        edges = logits
        dec_x, output, adj_A_tilt_decoder  = self.decoder(inputs, edges, inputs.size()[1], origin_A, adj_A_tilt_encoder, Wa)
        
        
        # only focus on reconstruction loss
        return output, edges