
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension. For binary prediction, out_features=1.
        L (int): Number of hidden layers.
    """

    def __init__(
        self,
        in_features: int = 384,
        out_features: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        L: int = 2,
        pooling = pyg_nn.pool.global_mean_pool,
        ):
        super(GraphHead, self).__init__()

        self.pooling_fun = pooling
        list_FC_layers = [
            nn.Linear(in_features // 2 ** l, in_features // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(in_features // 2 ** L, out_features, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = nn.SiLU()


    def forward(self, x, batch):
        graph_emb = self.pooling_fun(x, batch)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)

        return graph_emb

