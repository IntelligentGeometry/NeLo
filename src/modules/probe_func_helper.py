import numpy as np
import trimesh
import pymeshlab

import torch
from torch.nn import Linear, Parameter
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from torch_geometric.data import Data

from config.global_config import global_config




class FixedConv(MessagePassing):
    """
    A simple paramer-free convolutional layer with fixed weights. a layer of this kind of conv is 
    equivalent to multiplying the input with a fixed matrix.
    This is almost the same with https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SimpleConv.html
    """
    def __init__(self ):
        super().__init__(aggr='add')
        
    def forward(self, x, edge_index, edge_weight, debug=False):
        # x has shape [N, 1] or [N]
        # edge_index has shape [2, E]
        # edge_weight has shape [E, 1] or [E]
        # vertex_mass has shape [N, 1] or [N]. if None, then we assume all vertex mass is 1.
        
        if debug:
            assert edge_weight.shape[0] == edge_index.shape[1]      # make sure the number of edges is correct
            assert x.shape[0] == edge_index.max() + 1               # make sure the number of vertices is correct
        
        # Some other properties you may want to test:
        # 1. sum of edge_weight for a vertex is equal to 0
        # 2. every vertex has a self-loop, and only the self-loop has a positive weight
        # 3. apart from self-loop, every edge has a pair of opposite edge with the same weight

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        return out

    def message(self, x_i, x_j, edge_weight):
        # edge_weight: the weight of each edge. shape: either [E, 1] or [E] is OK
        # x_j has shape [E, out_channels]

        # Compute messages with fixed linear weight matrix.
        out = edge_weight.view(-1, 1) * x_j
        return out





