import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch_geometric
import torch.nn as nn



##################################################################
# Very basic modules
##################################################################

class SmallMLP(nn.Module):
    '''
    A simple MLP with 3 layers and Softplus (Not ReLU!) activation
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, bias=True):
        super(SmallMLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels, bias=bias)
        self.fc3 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.activation = torch.nn.functional.softplus

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x



class SmallGNN(nn.Module):
    def __init__(self, num_layer, input_dim, hidden_dim, output_dim):
        super(SmallGNN, self).__init__()
        # set the type of GNN layer
        conv_layer = torch_geometric.nn.SAGEConv
        # set the properties of the graph neural network
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        # build the graph neural network
        self.convs = nn.ModuleList()
        self.convs.append(conv_layer(input_dim, hidden_dim))
        for i in range(num_layer - 2):
            self.convs.append(conv_layer(hidden_dim, hidden_dim))
        self.convs.append(conv_layer(hidden_dim, output_dim))
        self.activation = torch.nn.functional.relu

    def forward(self, feature, graph):
        # feature: [num_nodes, input_dim]
        # data.x: [num_nodes, 3]
        # data.edge_index: [2, num_edges]
        x = feature
        for i in range(self.num_layer):

            x = self.convs[i](x, graph.edge_index)
            x = self.activation(x)
        #x = torch_geometric.nn.pool.global_mean_pool(x, batch=None)
        return x
    
    


class VertMassMLP(nn.Module):
    '''
    A simple MLP but modified for vertex mass prediction
    modification:
     - last layer has a softplus activation (so there won't be negative mass)
     - last linear has a bias of 0.7 (so the initial mass is around 1.0)
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, bias=True):
        super(VertMassMLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels, bias=bias)
        self.fc3 = nn.Linear(hidden_channels, out_channels, bias=bias)
        # initialize the last linear layer's bias
        self.fc3.bias.data.fill_(0.7)
        
        self.activation = torch.nn.functional.softplus
        self.softplus = torch.nn.functional.softplus

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = self.softplus(x)
        return x




from src.modules.probe_func_helper import FixedConv
from src.graphtree.data import Data
class SmoothingFixedConv(nn.Module):
    '''
    A simple smoother using FixedConv
    '''
    def __init__(self):
        super(SmoothingFixedConv, self).__init__()
        self.fixed_conv = FixedConv()

    def forward(self, graph: Data, 
                vertex_mass: torch.Tensor=None
                ):
        # NOTE: we assume that the graph do not have self-loop. Please make sure this!
        x = graph.x
        edge_index = graph.edge_index
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float32, device=edge_index.device)
        y = self.fixed_conv(x, edge_index, edge_weight)