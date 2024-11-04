import numpy as np
import torch
import torch_geometric
from src.graphtree.graph_tree import GraphTree
from src.graphtree.data import Data

from typing import Optional, Tuple


"""
Wrapper of graph laplacian
"""

def get_graph_laplacian(graph: Data,
                        normalization: Optional[str] = None,
                        ) :
    """
    Get the graph laplacian matrix of a graph.
    We first construct a KNN graph, then assign 1 to all the edges.
    graph: Data
    normalization: str or None. Could be None, 'sym' (symmetric normalization), 'rw' (random walk normalization)
    return: (edge_index, edge_weight). self-loop is included.
    """
    x = graph.x[:3]                      # [N, 3]
    edge_index = graph.edge_index        # [2, E]
    edge_weight = torch.ones(edge_index.shape[1], dtype=x.dtype, device=x.device)
    
    # normalize the edge_weight
    ret_edge_index, ret_edge_weight = torch_geometric.utils.get_laplacian(edge_index, edge_weight, normalization=normalization)
    
    # torch_geometric.utils.get_laplacian add self-loop to the graph automatically
    
    #         [2, E],           [E,]
    return ret_edge_index, ret_edge_weight


