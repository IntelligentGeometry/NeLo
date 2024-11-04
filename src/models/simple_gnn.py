import torch
import torch.nn as nn
from typing import Dict, Optional
import torch_geometric

from config.global_config import global_config


from src.graphtree.graph_tree import Data, GraphTree

import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)


"""
Simplest GNN model with no normalization, pooling, etc.
"""

class SimpleGNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int="default"):
        super(SimpleGNN, self).__init__()
        
        hidden_channels = global_config.gnn_hidden_channels
        
        if global_config.gnn_conv_type.lower() == 'sage':
            conv = torch_geometric.nn.SAGEConv
        elif global_config.gnn_conv_type.lower() == 'gcn':
            conv = torch_geometric.nn.GCNConv
        elif global_config.gnn_conv_type.lower() == 'gat':
            conv = torch_geometric.nn.GATv2Conv
        elif global_config.gnn_conv_type.lower() == 'edgeconv':
            conv = EdgeConv                       
        elif global_config.gnn_conv_type.lower() == 'gin':
            conv = torch_geometric.nn.GINConv       
        else:
            raise NotImplementedError(f"Unknown GNN conv type: {global_config.gnn_conv_type}")
        
        if num_layers == "default":
            num_layers = global_config.gnn_num_layers
            
        assert num_layers >= 2
        num_layers_middle = num_layers - 2
        
        self.convs = [ conv(in_channels, hidden_channels), ]
        for i in range(num_layers_middle):
            self.convs.append(conv(hidden_channels, hidden_channels))
        self.convs.append(conv(hidden_channels, out_channels))
        # move to cuda
        self.convs = nn.ModuleList(self.convs)


    def forward(self, x, graph_tree: GraphTree, depth):
        edge_index = graph_tree.treedict[0].edge_index
        # forward pass
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = torch.nn.functional.relu(x)
        # last layer
        x = self.convs[-1](x, edge_index)
        return x




