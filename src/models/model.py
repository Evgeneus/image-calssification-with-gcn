import torch
import torch.nn as nn
import dgl

from models.resnet import ResNet18
from models.graph_conv import GraphConv, WGraphConv
from models.contrastive import ContrastiveNet


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GraphResNet(nn.Module):

    def __init__(self, args):
        super(GraphResNet, self).__init__()

        # load or create encoder
        if args.pretrained_encoder is not None:
            self.encoder = ResNet18(output_layer=True)
            self.encoder.load_state_dict(torch.load(args.pretrained_encoder, map_location=args.device))
            self.encoder.linear = Identity()
        else:
            self.encoder = ResNet18(output_layer=False)
        embed_dim = 512  # output from ResNet18

        if args.pretrained_contrastive_model is not None:
            self.CoNet = ContrastiveNet(args)
            self.CoNet.load_state_dict(torch.load(args.pretrained_contrastive_model, map_location=args.device))
            for params in self.CoNet.parameters():
                params.requires_grad = False
        else:
            raise NotImplementedError("only support pretrained contrastive_model")

        self.graph_conv = WGraphConv(embed_dim, 10, bias=True)

    @torch.no_grad()
    def _get_grpah(self, x):
        # compute similarity scores, i e weights
        weights = self.CoNet.compute_similarity_matrix(x)
        # create densely connected weighted graph
        num_vert = x.shape[0]
        g = dgl.DGLGraph()
        g.add_nodes(num_vert)
        for i in range(num_vert):
            for j in range(num_vert):
                g.add_edge(i, j, {'weight': weights[i][j].unsqueeze(0).unsqueeze(1)})

        return g

    def forward(self, x):
        # make forward pass
        x1 = self.encoder(x)
        # create graph
        g = self._get_grpah(x)
        # run graph conv with weighted graph
        out = self.graph_conv(g, x1)

        return out


# from models.helper import GraphCreator
# from dgl.nn.pytorch import GATConv
# class ResNetGCN(nn.Module, GraphCreator):
#
#     def __init__(self, graph_type):
#         super(ResNetGCN, self).__init__()
#         embed_dim = 512  # output from ResNet18
#         self.graph_type = graph_type
#         self.encoder = ResNet18(output_layer=False)
#         self.graph_conv = GraphConv(embed_dim, embed_dim, bias=True)
#         self.fc = nn.Linear(embed_dim, 10, bias=True)
#
#     def forward(self, x):
#         # make forward pass
#         x1 = self.encoder(x)
#         # create graph
#         g = self._get_grpah(x1, self.graph_type)
#         x2 = self.graph_conv(g, x1)
#         out = self.fc(x1 + x2)
#
#         return out
#
#
# class ResNetGATCN(nn.Module, GraphCreator):
#
#     def __init__(self, graph_type):
#         super(ResNetGATCN, self).__init__()
#         embed_dim = 512  # output from ResNet18
#         self.graph_type = graph_type
#         self.encoder = ResNet18(output_layer=False)
#         self.gat_conv = GATConv(embed_dim, embed_dim, 1)
#         self.fc = nn.Linear(embed_dim, 10, bias=True)
#
#     def forward(self, x):
#         # make forward pass
#         x1 = self.encoder(x)
#         # create graph
#         g = self._get_grpah(x1, self.graph_type)
#
#         x2 = self.gat_conv(g, x1)
#         x2 = torch.squeeze(x2, 1)
#         out = self.fc(x1 + x2)
#
#         return out
