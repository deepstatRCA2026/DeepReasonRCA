import torch
import torch.nn as nn

import sys
sys.path.append('./')
from RCAEval.e2e.multimodel.inner_models.GTlayer import GTLayer

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class permute(nn.Module):
    def __init__(self):
        super(permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)

class GTN(nn.Module):
    def __init__(self, edge_types, num_channels, num_layers, norm):
        super(GTN, self).__init__()
        self.edge_types = edge_types
        self.num_channels = num_channels
        self.per = permute()
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(edge_types, num_channels, first=True))
            else:
                layers.append(GTLayer(edge_types, num_channels, first=False))
                
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(1, 1))
        self.loss = nn.CrossEntropyLoss()

    def normalization(self, H,device):
        for i in range(self.num_channels):
            if i == 0:
                H_ = self.norm(H[:, i, :, :], device).unsqueeze(1)
            else:
                H_ = torch.cat((H_, self.norm(H[:, i, :, :], device).unsqueeze(1)), dim=1)
        return H_


    def norm(self, H, device, add=True):
        if add == False:
            H = H * ((torch.eye(H.shape[1]) == 0).type(torch.FloatTensor)).unsqueeze(0)
        else:
            #H = H * ((torch.eye(H.shape[1]) == 0).type(torch.FloatTensor)).unsqueeze(0).to(device) + torch.eye(
            #    H.shape[1]).type(
            #    torch.FloatTensor).unsqueeze(0).to(device)
            #if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            #    device = torch.device(f'cuda:{torch.cuda.current_device()}')
            #else:
            #    device = torch.device('cpu')

            H = H.to(device)
            eye = torch.eye(H.shape[1], device=H.device)  # safest

            #eye = torch.eye(H.shape[1], device=device)  # already on GPU
            mask = (eye == 0).float().unsqueeze(0)      # [1, N, N], on GPU
            H = H * mask + eye.unsqueeze(0)             # elementwise multiply + add identity

        deg = torch.sum(H, dim=-1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv.view((deg_inv.shape[0], deg_inv.shape[1], 1)) * torch.eye(H.shape[1]).type(
            torch.FloatTensor).unsqueeze(0).to(device)
        H = torch.bmm(deg_inv, H)
        return H


    def forward(self, A,device):
        # A shape (B,N,N,C)
        A = A.unsqueeze(1).permute(0, 1, 4, 2, 3)
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H,device)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        return H
