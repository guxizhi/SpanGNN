import torch.nn as nn
import torch.nn.functional as F

import dgl.nn as dglnn
import dgl
import dgl.function as fn


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        return h


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=0., activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


class GNNModel(nn.Module):
    def __init__(self,  gnn_layer: str, n_layers: int, layer_dim: int,
                 input_feature_dim: int, n_classes: int, n_linear: int):
        super().__init__()

        assert n_layers >= 1, 'GNN must have at least one layer'
        dims = [input_feature_dim] + [layer_dim] * (n_layers-1) + [n_classes]
        print(dims)
        
        self.convs = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        self.n_layers = n_layers
        self.n_linear = n_linear
        
        for idx in range(n_layers):
            if idx < n_layers - n_linear:
                if gnn_layer == 'gat':
                    # use 2 aattention heads
                    # layer = dglnn.GATConv(dims[idx], dims[idx+1], 1)  # pylint: disable=no-member
                    layer = dglnn.AGNNConv(learn_beta=False, allow_zero_in_degree=True)
                elif gnn_layer == 'gcn':
                    layer = dglnn.GraphConv(dims[idx], dims[idx+1], allow_zero_in_degree=True)  # pylint: disable=no-member
                elif gnn_layer == 'sage':
                    # Use mean aggregtion
                    # pylint: disable=no-member
                    layer = dglnn.SAGEConv(dims[idx], dims[idx+1],
                                            aggregator_type='mean')
                else:
                    raise ValueError(f'unknown gnn layer type {gnn_layer}')
                self.convs.append(layer)
            else: 
                self.convs.append(nn.Linear(dims[idx], dims[idx+1]))
                
            if idx < n_layers - 1:
                self.norm.append(nn.LayerNorm(dims[idx+1], elementwise_affine=True))
                
            
    def forward(self, graph, features):
        h = features
        for idx in range(self.n_layers):
            
            h = F.dropout(h, p=0.5)
            if idx < self.n_layers - self.n_linear:
                h = self.convs[idx](graph, h)
                if h.ndim == 3:  # GAT produces an extra n_heads dimension
                    h = h.mean(1)
            else:
                h = self.convs[idx](h)

            if idx < self.n_layers - 1:
                h = self.norm[idx](h)
                h = F.relu(h, inplace=True)
            
        return h