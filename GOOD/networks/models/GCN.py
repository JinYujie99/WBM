r"""
The Graph Neural Network from the `"How Powerful are Graph Neural Networks?"
<https://arxiv.org/abs/1810.00826>`_ paper.
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
# from .GCN_double import GCNConv
from .MolEncoders import AtomEncoder, BondEncoder
from torch.nn import Identity


@register.model_register
class GCN(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = GCNFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        out_readout = self.feat_encoder(*args, **kwargs)
        out = self.classifier(out_readout)
        return out


class GCNFeatExtractor(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(GCNFeatExtractor, self).__init__(config)
        num_layer = config.model.model_layer
        self.encoder = MyGCNEncoder(config, **kwargs)

        if config.model.layered_feature:
            self.layered_feature = True
        else:
            self.layered_feature = False

    def forward(self, *args, **kwargs):
        if not self.layered_feature:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, batch, batch_size, **kwargs)
            return out_readout
        else:
            x, edge_index, batch, batch_size, ptr = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout, batch_embedded_features_uncat = self.encoder.forward_JK(x, edge_index, batch, batch_size, ptr, **kwargs)
            return out_readout, batch_embedded_features_uncat

class MyGCNEncoder(BasicEncoder):
    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):
        super(MyGCNEncoder, self).__init__(config, *args, **kwargs)
        num_layer = config.model.model_layer
        self.without_readout = kwargs.get('without_readout')

        self.conv1 = gnn.GCNConv(in_channels=config.dataset.dim_node, out_channels=config.model.dim_hidden)
        self.convs = nn.ModuleList(
            [
                gnn.GCNConv(in_channels=config.model.dim_hidden, out_channels=config.model.dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        self.conv1.lin = torch.nn.Linear(config.dataset.dim_node, config.model.dim_hidden, dtype=torch.float64)
        self.conv1.bias = torch.nn.Parameter(torch.zeros(config.model.dim_hidden, dtype=torch.float64))
        for i in range(num_layer-1):
            self.convs[i].lin = torch.nn.Linear(config.model.dim_hidden, config.model.dim_hidden, dtype=torch.float64)
            self.convs[i].bias = torch.nn.Parameter(torch.zeros(config.model.dim_hidden, dtype=torch.float64))



    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        x = x.to(torch.float64)

        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout

    def get_layered_feature(self, x, edge_index):
        x = x.to(torch.float64)
        layered_features = []
        layered_features.append(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        layered_features.append(post_conv)
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            layered_features.append(post_conv)

        batch_embedded_features = torch.cat(layered_features, dim=1)
        return batch_embedded_features

    def forward_JK(self, x, edge_index, batch, batch_size, ptr, **kwargs):

        x = x.to(torch.float64)
        layered_features = []
        layered_features.append(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        layered_features.append(post_conv)
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            layered_features.append(post_conv)

        batch_embedded_features = torch.cat(layered_features, dim=1)
        batch_embedded_features_uncat = [batch_embedded_features[ptr[k].item(): ptr[k + 1].item(), :] for k in range(len(ptr)-1)]

        if self.without_readout or kwargs.get('without_readout'):
             return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout, batch_embedded_features_uncat