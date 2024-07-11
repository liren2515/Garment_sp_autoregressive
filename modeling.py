import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from resnet import ResnetBlock

from resnet1D import ResNet1D

class testModel(nn.Module):
    def __init__(self, input_dim, out_dim, device):
                
        super().__init__()
        self.sage1 = torch_geometric.nn.SAGEConv(input_dim, 16)
        self.relu = nn.ReLU()
        self.sage2 = torch_geometric.nn.SAGEConv(16, 32)
        self.sage3 = torch_geometric.nn.SAGEConv(32, 64)
        self.sage4 = torch_geometric.nn.SAGEConv(64, 128)
        #self.sage5 = torch_geometric.nn.SAGEConv(128, 256)

        #self.resnet = ResNet1D(128, 64, 2, 1, 1, 12, out_dim)
        #self.resnet = ResNet1D(64, 64, 2, 1, 1, 8, out_dim)
        #self.resnet = ResNet1D(256, 64, 2, 1, 1, 16, out_dim)
        self.resnets = []
        res_list = [128, 128, 128, 128,
            192, 192, 192, 192,
            256, 256, 256, 256]
        cur = 128
        for dim_layer in res_list:
            self.resnets.append(ResnetBlock(cur, dim_layer, dropout = 0).to(device))
            cur = dim_layer
        self.L = nn.Linear(cur, out_dim)


    def forward(self, inputs, edge_index):
        out = self.sage1(inputs, edge_index)
        out = self.relu(out)
        out = self.sage2(out, edge_index)
        out = self.sage3(out, edge_index)
        out = self.sage4(out, edge_index)
        #out = self.sage5(out, edge_index)
        
        out = out.unsqueeze(2)
        for layer in self.resnets:
            out = layer(out)
        out = out.squeeze(2)
        out = self.L(out)

        return out