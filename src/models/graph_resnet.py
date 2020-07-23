import torch
import torch.nn as nn
import dgl
import numpy as np

from models.resnet import ResNet18
from models.graph_conv import WGraphConv
from models.contrastive import ContrastiveNet


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GraphResNet(nn.Module):

    def __init__(self, args):
        super(GraphResNet, self).__init__()
        self.args = args
        # load or create encoder for initial features of nodes
        if args.pretrained_encoder_gnn is not None:
            self.encoder = ResNet18(output_layer=True)
            self.encoder.load_state_dict(torch.load(args.pretrained_encoder_gnn, map_location=args.device))
            self.encoder.linear = Identity()
            for params in self.encoder.parameters():
                params.requires_grad = False
        else:
            self.encoder = ResNet18(output_layer=False)
        embed_dim = 512  # output from ResNet18

        # load pretrained contrastive model for computing similarity scores
        if args.pretrained_contrastive_model is not None:
            self.CoNet = ContrastiveNet(args)
            self.CoNet.load_state_dict(torch.load(args.pretrained_contrastive_model, map_location=args.device))
            for params in self.CoNet.parameters():
                params.requires_grad = False
        else:
            raise NotImplementedError("only support pretrained contrastive_model")

        self.num_classes = 10
        self.n = 10  # number of most similar nodes to consider for sending graph message
        self.graph_conv = WGraphConv(embed_dim, self.num_classes, bias=True)

    @torch.no_grad()
    def _get_grpah(self, x):
        # compute similarity scores, i e weights
        weights = self.CoNet.compute_similarity_matrix(x)
        # create densely connected weighted graph
        num_vert = x.shape[0]
        g = dgl.DGLGraph()
        g.add_nodes(num_vert)
        for i in range(num_vert):
            j_nodes = np.argpartition(weights[i].cpu().numpy(), -self.n)[-self.n:]
            for j in j_nodes:
                g.add_edge(i, j, {'weight': weights[i][j].unsqueeze(0).unsqueeze(1)})

        return g


    @torch.no_grad()
    def _get_grpah_softlabels(self, x, labels_hard):
        # compute similarity scores, i e weights
        weights = self.CoNet.compute_similarity_matrix(x)
        # create densely connected weighted graph
        num_vert = x.shape[0]
        g = dgl.DGLGraph()
        g.add_nodes(num_vert)
        labels_soft = []
        for t_node in range(num_vert):
            # compute soft labels
            source_nodes = np.argpartition(weights[t_node].cpu().numpy(), -self.n)[-self.n:]
            labels_source_nodes = labels_hard[source_nodes]
            sim_source_nodes = weights[t_node][source_nodes]
            soft = torch.zeros(10)
            norm = sim_source_nodes.sum().item()
            for l, sim in zip(labels_source_nodes, sim_source_nodes):
                soft[l] += sim
            soft /= norm
            labels_soft.append(soft)
            # create weighted edges for graph
            for s_node in source_nodes:
                w = weights[t_node][s_node].unsqueeze(0).unsqueeze(1)
                g.add_edge(s_node, t_node, {'weight': w})

        labels_soft = torch.stack(labels_soft, dim=0).long().to(self.args.device)
        g.to(self.args.device)
        return g, labels_soft

    def forward(self, x, labels_hard=None):
        # make forward pass
        x1 = self.encoder(x)
        # create graph/soft labels
        if labels_hard is None:
            g = self._get_grpah(x)
        else:
            g, labels_soft = self._get_grpah_softlabels(x, labels_hard)
        # run graph convolution
        out = self.graph_conv(g, x1)

        if labels_hard is None:
            return out
        else:
            return out, labels_soft

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
