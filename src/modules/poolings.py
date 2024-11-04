import numpy as np
import trimesh
import pymeshlab
from config.global_config import global_config

import torch
import torch_geometric
import torch.nn as nn

from torch_geometric.nn import fps
from torch_geometric.nn.unpool import knn_interpolate


'''
This file is a simple implementation of the pooling and unpooling operations in PointNet++.
Maybe in the future we will have a better one.
'''


class PoolingGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data: torch_geometric.data.Data, mesh: list[trimesh.Trimesh]):
        assert False, "legacy code, not used anymore"
        pass
        
        

class UnpoolingGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, coarse_feature: torch.tensor, 
                coarse_data: torch_geometric.data.Data, 
                fine_data: torch_geometric.data.Data):
        assert False, "legacy code, not used anymore"
        
        
