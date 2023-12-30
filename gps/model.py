import torch
from gps.layers import GPSLayer
from gps.encoders import FeatureEncoder
import torch.nn as nn

from torch_geometric.utils import to_dense_batch


class GPSEncoder(nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(
        self,
        d_model = 384,
        nhead = 16,
        dropout = 0.1,
        attention_dropout = 0.1,
        layer_norm = False,
        batch_norm = True,
        momentum = 0.9,
        log_attention_weights = True,
        num_layer = 10,
        ):
        super(GPSEncoder, self).__init__()

        self.encoder = FeatureEncoder()

        self.gps = nn.ModuleList(
            [GPSLayer(
                d_model,
                nhead,
                dropout,
                attention_dropout,
                layer_norm,
                batch_norm,
                momentum,
                log_attention_weights,
            ) for _ in range(num_layer-1)
            ]+[GPSLayer(
                d_model,
                nhead,
                dropout,
                attention_dropout,
                layer_norm,
                batch_norm,
                momentum,
                log_attention_weights,
                last=True)
                ]
        )


    def forward(
        self,
        batch

        ):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        rwse = batch.rwse
        batch = batch.batch
        x, edge_attr = self.encoder(x, edge_attr, rwse)

        for layer_idx, enc_layer in enumerate(self.gps):
            last = True if layer_idx == len(self.gps) -1 else False
            x, edge_attr = enc_layer(x, edge_index, edge_attr, batch, last=last)
        
        return x, batch