import torch
import torch.nn as nn
from torch.nn import init


class GraphConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 activation=None):
        super(GraphConv, self).__init__()

        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def _gcn_message(self, edges):
        return {'msg': edges.src['h']}

    def _gcn_reduce(self, nodes):
        return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}

    def forward(self, graph, feat):

        graph = graph.local_var()

        # compute norm
        degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp)
        feat = feat * norm

        graph.ndata['h'] = feat
        graph.update_all(self._gcn_message, self._gcn_reduce)
        rst = graph.ndata.pop('h')

        weight = self.weight
        rst = torch.matmul(rst, weight)

        degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp)
        rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst


class WGraphConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 activation=None):
        super(WGraphConv, self).__init__()

        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def _gcn_message(self, edges):
        return {'msg':  edges.data['weight'] * edges.src['h']}

    def _gcn_reduce(self, nodes):
        return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}

    def forward(self, graph, feat):

        graph = graph.local_var()

        # compute norm
        degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp)
        feat = feat * norm

        graph.ndata['h'] = feat
        graph.update_all(self._gcn_message, self._gcn_reduce)
        rst = graph.ndata.pop('h')

        weight = self.weight
        rst = torch.matmul(rst, weight)

        degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp)
        rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst
