from sklearn.cluster import KMeans
import numpy as np
import dgl
import torch


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
