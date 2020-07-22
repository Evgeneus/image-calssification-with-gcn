import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GATConv

from models.resnet import ResNet18
from models.helper import GraphCreator
from models.graph_conv import GraphConv, WGraphConv
from models.nt_xent import ContrastiveNet


class ResNetGCN(nn.Module, GraphCreator):

    def __init__(self, graph_type):
        super(ResNetGCN, self).__init__()
        embed_dim = 512  # output from ResNet18
        self.graph_type = graph_type
        self.encoder = ResNet18(output_layer=False)
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
        embed_dim = 512  # output from ResNet18
        self.graph_type = graph_type
        self.encoder = ResNet18(output_layer=False)
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


class ContrastiveResNetGCN(nn.Module):

    def __init__(self, graph_type, batch_size, temp, device, projection_dim):
        super(ContrastiveResNetGCN, self).__init__()
        embed_dim = 512  # output from ResNet18
        self.graph_type = graph_type
        self.encoder = ResNet18(output_layer=False)

        # We use a MLP with one hidden layer to obtain projection z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim, bias=False),
        )

        self.contrastive_net = ContrastiveNet(batch_size, temp, device)

        self.graph_conv = WGraphConv(embed_dim, 128, bias=True)
        self.fc = nn.Linear(128, 10, bias=True)

    def _cosine_similarity(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)

        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    @torch.no_grad()
    def _get_grpah(self, x_proj):
        weights = self._cosine_similarity(x_proj)
        num_vert = x_proj.shape[0]
        g = dgl.DGLGraph()
        g.add_nodes(num_vert)
        for i in range(num_vert):
            for j in range(num_vert):
                g.add_edge(i, j, {'weight': weights[i][j].unsqueeze(0).unsqueeze(1)})
        print(weights[0].detach().numpy())
        print("--------------")
        return g

    def forward(self, x, targets=None):
        # make forward pass
        x1 = self.encoder(x)

        # similarity scores
        x_proj = self.projector(x1)
        if targets is not None:
            loss_contrast = self.contrastive_net(x_proj, targets)

        # create graph
        g = self._get_grpah(x_proj)
        out = self.graph_conv(g, x1)

        if targets is not None:
            return out, loss_contrast
        else:
            return out
