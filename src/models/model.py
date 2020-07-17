import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv

from models.resnet import ResNet18
from models.helper import GraphCreator
from models.graph_conv import GraphConv


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        embed_dim = 100
        self.encoder = ResNet18(num_classes=embed_dim)
        self.fc = nn.Linear(embed_dim, 10, bias=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)

        return x


class ResNetGCN(nn.Module, GraphCreator):

    def __init__(self, graph_type):
        super(ResNetGCN, self).__init__()
        embed_dim = 100
        self.graph_type = graph_type
        self.encoder = ResNet18(num_classes=embed_dim)
        self.graph_conv = GraphConv(embed_dim, embed_dim, bias=True)
        self.fc = nn.Linear(embed_dim, 10, bias=True)

    def forward(self, x):
        # make forward pass
        x1 = self.encoder(x)
        # create graph
        g = self._get_grpah(x1, self.graph_type)
        x2 = self.graph_conv(g, x1)
        out = self.fc(x1 + x2)

        return out


class ResNetGATCN(nn.Module, GraphCreator):

    def __init__(self, graph_type):
        super(ResNetGATCN, self).__init__()
        embed_dim = 100
        self.graph_type = graph_type
        self.encoder = ResNet18(num_classes=embed_dim)
        self.gat_conv = GATConv(embed_dim, embed_dim, 1)
        self.fc = nn.Linear(embed_dim, 10, bias=True)

    def forward(self, x):
        # make forward pass
        x1 = self.encoder(x)
        # create graph
        g = self._get_grpah(x1, self.graph_type)

        x2 = self.gat_conv(g, x1)
        x2 = torch.squeeze(x2, 1)
        out = self.fc(x1 + x2)

        return out
