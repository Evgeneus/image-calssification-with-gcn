import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import dgl
from dgl.nn.pytorch import GATConv, GraphConv
from sklearn.cluster import KMeans
import numpy as np


# class MLP(nn.Module):
#
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#
#         return x
#
#
# class GATConvMLP(nn.Module):
#
#     def __init__(self):
#         super(GATConvMLP, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.gat_conv = GATConv(84, 10, 1)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#
#         # create densely connected graph
#         num_vert = x.shape[0]
#         src = []
#         for i in range(num_vert):
#             s = [i for _ in range(num_vert)]
#             src += s
#         dst = [i for _ in range(num_vert) for i in range(num_vert)]
#         src = torch.tensor(src)
#         dst = torch.tensor(dst)
#         g = dgl.graph((src, dst))
#
#         h = self.gat_conv(g, x)
#         h = torch.squeeze(h, 1)
#
#         return h
#
#
# class GConvMLP(nn.Module):
#
#     def __init__(self):
#         super(GConvMLP, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.graph_conv = GraphConv(84, 10, bias=True)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#
#         # create densely connected graph
#         num_vert = x.shape[0]
#         src = []
#         for i in range(num_vert):
#             s = [i for _ in range(num_vert)]
#             src += s
#         dst = [i for _ in range(num_vert) for i in range(num_vert)]
#         src = torch.tensor(src)
#         dst = torch.tensor(dst)
#         g = dgl.graph((src, dst))
#
#         h = self.graph_conv(g, x)
#         h = torch.squeeze(h, 1)
#
#         return h

class GraphCreator:

    def _get_grpah(self, x, graph_type):
        src, dst = [], []
        if graph_type == "FC":
            # create densely connected graph
            num_vert = x.shape[0]
            for i in range(num_vert):
                for j in range(num_vert):
                    src.append(i)
                    dst.append(j)
        elif graph_type == "K-MEAN":
            # create graph using k-means to form isolated subgraphs
            kmeans = KMeans(n_clusters=10).fit(x.detach().numpy())
            labels = kmeans.labels_
            labels_set = set(labels)
            for l in labels_set:
                vertx = np.where(labels == l)[0]
                for i in vertx:
                    for j in vertx:
                        src.append(i)
                        dst.append(j)
        elif graph_type == "Empty":
            num_vert = x.shape[0]
            src = list(range(num_vert))
            dst = list(range(num_vert))
        else:
            raise NotImplementedError("graph type not implemented")

        src = torch.tensor(src)
        dst = torch.tensor(dst)
        g = dgl.graph((src, dst))

        return g


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        embed_dim = 100
        self.encoder = models.resnet18(num_classes=embed_dim)
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
        self.encoder = models.resnet18(num_classes=embed_dim)
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
        self.encoder = models.resnet18(num_classes=embed_dim)
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
