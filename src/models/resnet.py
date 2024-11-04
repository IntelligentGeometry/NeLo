import torch
import torch.nn as nn
from typing import Dict, Optional
import torch_geometric


from src.graphtree.graph_tree import Data
from src.graphtree.graph_tree import GraphTree

from src.modules.resblocks import *

from src.modules.blocks import *







class MyMLP(torch.nn.Module):
    r''' 
        ResNet for graph data, with vertex-wise output
    '''

    def __init__(self, in_channels: int, out_channels: int): 
        super(MyMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear_1 = nn.Linear(in_channels, 32)
        self.linear_2 = nn.Linear(32, 32)
        self.linear_3 = nn.Linear(32, 32)
        self.linear_4 = nn.Linear(32, out_channels)


    def forward(self, data: torch.Tensor, graphtree: GraphTree, depth: int):
        """_summary_
        """
        x = self.linear_1(data)
        x = nn.functional.relu(x)
        x = self.linear_2(x)
        x = nn.functional.relu(x)
        x = self.linear_3(x)
        x = nn.functional.relu(x)
        x = self.linear_4(x)
        
        return x



class GraphResNetVertexWiseOutput(torch.nn.Module):
    r''' 
        ResNet for graph data, with vertex-wise output
    '''

    def __init__(self, in_channels: int, out_channels: int): 
        super(GraphResNetVertexWiseOutput, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear_1 = nn.Linear(in_channels, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.linear_3 = nn.Linear(128, 128)
        self.linear_4 = nn.Linear(128, out_channels)
        self.resblk_1 = GraphResBlocks(128, 128, resblk_num=2)
        self.resblk_2 = GraphResBlocks(128, 128, resblk_num=2)
        self.resblk_3 = GraphResBlocks(128, 128, resblk_num=2)


    def forward(self, data: torch.Tensor, graphtree: GraphTree, depth: int):
        """_summary_
        """
        x = self.linear_1(data)
        x = nn.functional.relu(x)
        x = self.resblk_1(x, graphtree, depth)
        x = self.resblk_2(x, graphtree, depth)
        x = self.resblk_3(x, graphtree, depth)
        x = self.linear_4(x)
        
        return x
  
  
  
class GraphResNetGraphWiseOutput(torch.nn.Module):
    r''' 
        ResNet for graph data, with graph-wise output
    '''

    def __init__(self, in_channels: int, out_channels: int): 
        super(GraphResNetGraphWiseOutput, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear_1 = nn.Linear(in_channels, 64)
        self.resblk_1 = GraphResBlocks(64, 64, resblk_num=2)
        self.resblk_2 = GraphResBlocks(64, 64, resblk_num=4)
        self.resblk_3 = GraphResBlocks(64, 64, resblk_num=2)
        self.linear_2 = nn.Linear(64, 64)
        self.linear_3 = nn.Linear(64, 64)
        self.Linear_4 = nn.Linear(64, out_channels)

    def forward(self, data: torch.Tensor, graphtree: GraphTree, depth: int):
        """_summary_
        """
        x = self.linear_1(data)
        x = self.resblk_1(x, graphtree, depth)
        x = self.resblk_2(x, graphtree, depth)
        x = self.resblk_3(x, graphtree, depth)
        x = self.linear_2(x)
        
        # extract global features
        x = torch_geometric.global_mean_pool(x,)
        # finally, output
        x = self.linear_3(x)
        x = torch.relu(x)
        x = self.linear_4(x)
        
        return x
    
    
    
    