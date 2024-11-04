import numpy as np
import trimesh
import pymeshlab

import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from torch_geometric.data import Data

from config.global_config import global_config

from src.modules.probe_func_helper import FixedConv
from src.graphtree.graph_tree import GraphTree




class ProbeFunction(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.probe_channel = global_config.probe_function_channels
        
        # fix-conv
        self.fixed_conv = FixedConv()
        
        # freq
        self.freqs = torch.logspace(0, global_config.probe_function_channels//2-0.5, 
                                    global_config.probe_function_channels,          # frequecy increases by sqrt(2)
                                    base=2, dtype=torch.float).cpu()
        
        #self.freqs = torch.linspace(1, global_config.probe_function_channels//2, 
         #                           global_config.probe_function_channels, 
          #                          dtype=torch.float).cpu()
        
    
    def forward(self, 
                data: Data, 
                probe: torch.Tensor, 
                edge_width: torch.Tensor, 
                automatic_add_self_loop: bool=True,
                in_test: bool=False):
        """
        This function is a combination of get_probe and lap_times_probe.
        However, it is not recommended to use this function, because it is not 
        efficient: it will calculate the probe at each call.
        """
        
        probe = self.get_probe(data, in_test)
        after_lap = self.lap_times_probe(data, probe, edge_width, automatic_add_self_loop)
        return after_lap
        
        
        
    def lap_times_probe(self, 
                        data: Data, 
                        probe: torch.Tensor, 
                        edge_width: torch.Tensor, 
                        vertex_mass: torch.Tensor,
                        vertex_mass_inverse: bool=True,
                        automatic_add_self_loop: bool=True,
                        normalize_by_degree: bool=False,
                        ignore_error_and_return_anyway: bool=False
                        ):
        """
        Given a graph and a Laplacian matrix represented by edge weights, calculate the result of the probe function after the Laplacian operator.
        data: a graph that has similar structure as torch_geometric.data.Data.
        probe: probe function with shape: [N, probe_channel]
        edge_width: edge weights with shape: [E, 1]
        vertex_mass: vertex mass with shape: [N, 1]
        automatic_add_self_loop: whether to add self loop to the graph. If the graph already has self loop, this should be False.
        normalize_by_degree: deprecated. 
        return: [N, probe_channel]
        ignore_error_and_return_anyway: if True, the function will return anyway even if the input vertex_mass is None. This parameter is only for testing. It is not recommended to use.
        """
        
        if data is None:
            if ignore_error_and_return_anyway:
                return torch.zeros(probe.shape[0], probe.shape[1], device="cuda")
            else:
                raise ValueError("The input graph (stiffness matrix) should not be None.")
        
        # assert vertex_mass is not None, and make its shape [N, 1]
        assert vertex_mass is not None
        assert vertex_mass.shape[0] == data.x.shape[0]
        if vertex_mass.shape.__len__() == 1:
            vertex_mass = vertex_mass.view(-1, 1)
                
        if automatic_add_self_loop:
            # test if the graph already has self loop
            # if (data.edge_index[0]==data.edge_index[1]).sum() > 0:
            #   raise ValueError("The graph already has self loop.")
            
            # add self loop
            self_loop_edge_index = torch.stack([torch.arange(data.x.shape[0], device="cuda"), torch.arange(data.x.shape[0], device="cuda")], dim=0)
            new_edge_index = torch.cat([data.edge_index, self_loop_edge_index], dim=1)
            # get the weight of self loop
            edge_weight_sum = torch.zeros(data.x.shape[0], 1, device="cuda")
            old_edge_index = data.edge_index
            edge_weight_sum.index_add_(0, old_edge_index[0], -edge_width)
            #for i in range(old_edge_index.shape[1]):
            #   edge_weight_sum[old_edge_index[0, i]] += -edge_width[i]
                #edge_weight_sum[data.edge_index[1, i]] += -edge_width[i]  
            new_edge_width = torch.cat([edge_width, edge_weight_sum], dim=0)
        else:
            new_edge_index = data.edge_index
            new_edge_width = edge_width
        
        # the probe function is multiplied by the Laplacian matrix, which is equivalent to a propagation of the probe function on the graph.
        after_lap = self.fixed_conv(probe, new_edge_index, new_edge_width)
        # apply mass
        if vertex_mass_inverse:
            after_lap = (1 / vertex_mass) * after_lap
        else:
            after_lap = vertex_mass * after_lap
        
        if normalize_by_degree:
            raise NotImplementedError("This function is not implemented yet.")

        
        return after_lap
    
    
        
    def get_probe(self, data: Data, in_test: bool=False, ref_graph_tree: GraphTree=None):        
        """
        data: Data used to generate the probe functions. Specifically, we primarily use data.x (the xyz coordinates of points in the data) to generate the special probe.
            Note that for the spectral probe, eigenvectors from the underlying GT graph are used.
            Specifically, for noisy training/testing, this data can be either a noisy graph or the original graph. The two have slightly different implications, though neither may be entirely reasonable. We leave it to future researchers to find a better metric.
        in_test: Indicates whether we are currently on the test set or the training set. If on the test set, the type of probe function might differ.
        ref_graph_tree: The reference graph tree, which should include one.
        """

        
        if in_test == False:
            probe_type = global_config.probe_function_type
        else:
            probe_type = global_config.val_probe_function_type
            
        probe_type = [each.lower() for each in probe_type]
        probe_outputs = []
        probe_channel = self.probe_channel
        freqs = self.freqs.cuda()
        
        """
        tri
        """
        
            
        if "tri_random" in probe_type:
            # weight of x, y, z, and normalize them
            x_weight = torch.rand(1, 1).cuda()
            y_weight = torch.rand(1, 1).cuda()
            z_weight = torch.rand(1, 1).cuda()
            x_weight = x_weight / (x_weight + y_weight + z_weight)
            y_weight = y_weight / (x_weight + y_weight + z_weight)
            z_weight = z_weight / (x_weight + y_weight + z_weight)
            # random phase between 0 and 2pi
            phase = torch.rand(1, 1).cuda() * 2 * np.pi
            # the noise on the frequency, between 1/sqrt(2) and sqrt(2) 
            freq_noise = torch.rand(1, 1).cuda() * 0.5 + 0.75
            # construct the probe
            coord = data.x[:, :3]
            coord_sum = coord[:, 0:1] * x_weight + coord[:, 1:2] * y_weight + coord[:, 2:3] * z_weight
            coord_sum = coord_sum.view(-1, 1)       # shape: [N, 1]
            freqs = freqs * freq_noise              # shape: [1, probe_channel//2]
            probe_input = coord_sum * freqs + phase # shape: [N, probe_channel//2]
            probe_output = [0.5/freqs**1 * torch.sin(probe_input), 0.5/freqs**1 * torch.cos(probe_input)]
            probe_outputs.extend(probe_output)
            
        """
        20240112
        """
        
    
        
        if "tri_xyz_test" in probe_type:
            test_freqs = torch.logspace(0, 14//2-1, 
                                        7,          # 这样频率以二的倍率递增
                                        base=2, dtype=torch.float).cpu()
            test_freqs = test_freqs.view(1, -1).cuda()      # shape: [1, 8]
            coord_x = data.x[:, 0].view(-1, 1)              # shape: [N, 1]
            coord_y = data.x[:, 1].view(-1, 1)              # shape: [N, 1]
            coord_z = data.x[:, 2].view(-1, 1)              # shape: [N, 1]
            probe_input_x = coord_x * test_freqs            # shape: [N, 8]
            probe_input_y = coord_y * test_freqs            # shape: [N, 8]
            probe_input_z = coord_z * test_freqs            # shape: [N, 8]
            probe_output_x = [0.5/test_freqs * torch.sin(probe_input_x), 0.5/test_freqs * torch.cos(probe_input_x)]     # shape: [N, 16]
            probe_output_y = [0.5/test_freqs * torch.sin(probe_input_y), 0.5/test_freqs * torch.cos(probe_input_y)]
            probe_output_z = [0.5/test_freqs * torch.sin(probe_input_z), 0.5/test_freqs * torch.cos(probe_input_z)]
            # finally: [N, 16*3] = [N, 48]
            probe_outputs.extend(probe_output_x)
            probe_outputs.extend(probe_output_y)
            probe_outputs.extend(probe_output_z)
            
            
        
            
        if "xyz_test" in probe_type:
            coord = data.x[:, :3]       # shape: [N, 3]
            A = coord                   # shape: [N, 3]
            B = coord**2                # shape: [N, 3]
            probe_output = [A, B]
            probe_outputs.extend(probe_output)
        
        
            
        """
        2024.5.1 some manual designed probe functions
        """
        def handcraft_probe_1(data: Data, scale_factor: int):
            """
            A handcraft probe function, shaped like a ripple, spreading from the center.
            """
            coord = data.x[:, :3] / scale_factor
            X = coord[:, 0].view(-1, 1)
            Y = coord[:, 1].view(-1, 1)
            Z = coord[:, 2].view(-1, 1)
            temp = torch.sqrt(X**2 + Y**2 + Z**2) 
            temp = torch.sin(temp * 25)
            return temp
        
        if "handcraft_1" in probe_type:
            probe_output = [handcraft_probe_1(data, 1), handcraft_probe_1(data, 1.927), handcraft_probe_1(data, 3.8154)]
            probe_outputs.extend(probe_output)
            
        def handcraft_probe_2(data: Data):
            """
            
            """
            coord = data.x[:, :3]
            X = coord[:, 0].view(-1, 1)
            Y = coord[:, 1].view(-1, 1)
            Z = coord[:, 2].view(-1, 1)

            temp = torch.sin(12*Z+2*X)
            temp = temp.view(-1, 1)
            #temp = torch.tensor(temp, dtype=torch.float).cuda()
            return temp
        
        if "handcraft_2" in probe_type:
            probe_output = [handcraft_probe_2(data)]
            probe_outputs.extend(probe_output)
            


        
        if "const" in probe_type:
            """
            Constant (all-one) probe function
            1. For each input vertex, generate an all-ones vector of length probe_channel.
            2. That’s it.
            Comments: Due to the properties of the Laplacian matrix, the output of this probe function is clearly all zeros. Therefore, the sole purpose of this probe is to test whether the properties of the Laplacian matrix are correctly implemented.
            """

            probe_output = torch.ones(data.x.shape[0], 1).cuda()
            probe_outputs.append(probe_output)
            
       
            
        if "heat_source" in probe_type:
            """
            not used in the paper
            """
            # randomly select some points as heat source
            # note that here data.x.shape[0] is the number of ALL vertices in the batch
            heat_source_index = torch.randint(0, data.x.shape[0], (data.x.shape[0]//300,)).cuda()
            probe_output = torch.zeros(data.x.shape[0], 1).cuda()
            probe_output[heat_source_index] = 100
            probe_outputs.append(probe_output)
            
        if "eigen" in probe_type:
            """
            Get the probe function from the eigenvectors of the graph.
            """
            start_idx = global_config.probe_eigen_start
            end_idx = global_config.probe_eigen_end
            probe_output = ProbeFunction.get_probe_from_eigen(ref_graph_tree, 
                                                              start_idx=start_idx,
                                                              end_idx=end_idx,
                                                              ) 
            probe_outputs.append(probe_output)
            
        
        return torch.cat(probe_outputs, dim=1)     # shape: [N, probe_channel]
            
            
    @staticmethod
    def get_probe_from_eigen(graph_tree: GraphTree,
                             normalize_by_eigen_value: bool=True,
                             start_idx: int=None,
                             end_idx: int=None,
                             ) -> torch.Tensor:
        """
        Generate probe function from the graph's eigenvectors
        normalize_by_eigen_value: Whether to divide each eigenvector by its corresponding eigenvalue. This way, eigenvectors with smaller eigenvalues will be "amplified," preventing them from being overly ignored during training due to their small values.
        start_idx: The starting index of the eigenvectors. If None, starts from the first eigenvector.
        end_idx: The ending index of the eigenvectors. If None, ends at the last eigenvector.
        """

        
        num_samples = graph_tree.batch_size
        all_probes = []

        for i in range(num_samples):
            probe = graph_tree.layers_eigen_vecs[0][i]
            if normalize_by_eigen_value:
                eigen_val = graph_tree.layers_eigen_vals[0][i]
                eigen_val = eigen_val.view(1, -1)
                probe = probe / (eigen_val + 0.1)    
            all_probes.append(probe)
        all_probes = torch.cat(all_probes, dim=0)
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = all_probes.shape[1]
        all_probes = all_probes[:, start_idx:end_idx]
        return all_probes
        
        
        
    @staticmethod
    def add_self_loop(data: Data, edge_width: torch.Tensor) -> (Data, torch.Tensor):
        """
        same functionality as a part of self.lap_times_probe
        """
        
        self_loop_edge_index = torch.stack([torch.arange(data.x.shape[0], device="cuda"), torch.arange(data.x.shape[0], device="cuda")], dim=0)
        new_edge_index = torch.cat([data.edge_index, self_loop_edge_index], dim=1)
        # get the weight of self loop
        edge_weight_sum = torch.zeros(data.x.shape[0], 1, device="cuda")
        old_edge_index = data.edge_index
        edge_weight_sum.index_add_(0, old_edge_index[0], -edge_width)
        
        new_edge_width = torch.cat([edge_width, edge_weight_sum], dim=0)
        new_data = Data(x=data.x, edge_index=new_edge_index)
        
        return new_data, new_edge_width
    
    @staticmethod
    def add_self_loop_in_matrix(matrix, copy=True):
        """
        For each matrix, set its diagonal elements to the sum of each row/column: M[i, i] = -sum(M[i, :])
        Note that this matrix should be symmetric, so sum(M[i, :]) = sum(M[:, i]).
        """

        if copy == False:
            raise NotImplementedError("This function does not support in-place operation.")
        if type(matrix) == torch.Tensor:
            matrix_new = matrix.clone()
        elif type(matrix) == np.ndarray:
            matrix_new = matrix.copy()
            
        assert matrix.shape[0] == matrix.shape[1]
        assert matrix[:, 1].sum() == matrix[1, :].sum()
        
        for i in range(matrix.shape[0]):
            matrix_new[i, i] = - matrix[i, :].sum()
        return matrix_new
        
        