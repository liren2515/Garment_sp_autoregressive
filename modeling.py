import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from resnet1D import ResNet1D

class testModel(nn.Module):
    def __init__(self, input_dim):
                
        super().__init__()
        self.sage1 = torch_geometric.nn.SAGEConv(input_dim, 16)
        self.relu = nn.ReLU()
        self.sage2 = torch_geometric.nn.SAGEConv(16, 32)

        self.resnet = ResNet1D(32, 64, 2, 1, 1, 8, input_dim)

    def forward(self, inputs, edge_index):
        out = self.sage1(inputs, edge_index)
        out = self.relu(out)
        out = self.sage2(out, edge_index)
        
        out = out.unsqueeze(2)
        out = self.resnet(out)
        return out