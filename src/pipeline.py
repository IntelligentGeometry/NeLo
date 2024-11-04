from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch_geometric
import torch.nn as nn
import pytorch_lightning as pl
import trimesh
import time

import matplotlib.pyplot as plt

from src.modules.basic import SmallMLP, VertMassMLP


from src.models.unet import GraphUNet
from src.models.resnet import GraphResNetVertexWiseOutput, MyMLP
from src.models.simple_gnn import SimpleGNN

# pooling and unpooling
#from src.modules.poolings import PoolingGraph
#from src.modules.poolings import UnpoolingGraph

# small utils
from config.global_config import global_config, console
import visualization.littlerender_binding as littlerender_binding


import src.loss_funcs as loss_funcs
from src.modules.probe import ProbeFunction

from src.graphtree.data import Data


import robust_laplacian
from collections import defaultdict
from scipy.sparse import coo_matrix


###############################################################
# Our Pipeline of eigen predicting
###############################################################

class MyPipeline(pl.LightningModule):
    def __init__(self, ):
        
        super(MyPipeline, self).__init__()
        
        # backup of history loss
        self.history_train_loss = defaultdict(list)
        
        # define the loss
        self.vertexwise_loss_func = loss_funcs.vertexwise_loss
     #   self.edgewise_loss_func = loss_funcs.edgewise_loss
        
        # training-specific parameters
        self.max_batch_size = global_config.batch_size
        self.batch_size = -1
        self.epochs = global_config.epochs
        
        # define the network and optimizer
        self.optimizer_name = global_config.optimizer
        self.learning_rate = global_config.learning_rate
        self.weight_decay = global_config.weight_decay
        self.scheduler_name = global_config.scheduler
        
        # debugging things
        self.per_channel_loss = dict()
        self.how_many_batches_in_this_epoch = 0
        
        # define the skeleton network
        if global_config.model == "unet":
            self.eigen_network = GraphUNet(in_channels=global_config.input_channels,
                                           out_channels=global_config.embedding_dim,
                                           )
        elif global_config.model == "resnet":
            self.eigen_network = GraphResNetVertexWiseOutput(in_channels=global_config.input_channels, 
                                                             out_channels=global_config.embedding_dim
                                                             )
        elif global_config.model == "gnn":
            self.eigen_network = SimpleGNN(in_channels=global_config.input_channels,
                                             out_channels=global_config.embedding_dim,
                                             )
            
        elif global_config.model == "mlp":
            self.eigen_network = MyMLP(in_channels=global_config.input_channels, 
                                        out_channels=global_config.embedding_dim
                                        )
        else:
            raise NotImplementedError("network not implemented")
        
        # define a decoder to decode the vertex-wise feature
     #   self.vert_decoder = SmallMLP(global_config.embedding_dim, global_config.embedding_dim*2, global_config.embedding_dim)
        
        # define a decoder to decode the vertex-wise mass (M in laplacian-beltrami)
        if global_config.use_vertex_mass == True:
            self.mass_decoder = VertMassMLP(global_config.embedding_dim, global_config.embedding_dim*2, 1, bias=True)
        
        # define a decoder to decode the edge-wise weight (L in laplacian-beltrami)
        self.edge_decoder = SmallMLP(global_config.embedding_dim, global_config.embedding_dim*2, 1, bias=False)     # 应该可以改成True。待后期测试
        
        # probe func module (parameter free)
        self.probe_func = ProbeFunction()
        
        
    
    def fake_make_prediction(self, batch: dict, 
                        batch_idx: int, 
                        compare_with_others: bool = False,
                        in_test=False):
        #time.sleep(0.2)
        loss_dict = {}
        x = batch['graph_tree'].cuda()
        x = x.treedict[0].x
        
        loss_dict["loss_total"] = x
        return loss_dict
    
    
    ######################################################################
    # training
    ######################################################################
    
    #@profile
    def make_prediction(self, 
                        batch: dict, 
                        batch_idx: int, 
                        compare_with_others: bool = False,
                        in_test=False):
        
        """
        get a graph, and predict the laplacian 
        """

        # get the data
        graph_tree = batch['graph_tree']#.cuda()
        self.batch_size = graph_tree.batch_size
        if in_test:
            in_train = False
        else:
            in_train = True
        
        ###################################################################
        # Pipeline:
        #
        # 1. Forwards the input graph, and predict vertexwise feature
        # 2. Combine two vertexwise features to get edgewise feature
        # 3. Compare the predicted edge result with the ground truth
        #
        ###################################################################
        
        ###########################################
        # STEP 1: Forwards the input graph, and predict vertexwise feature
        ###########################################
 
        # get input feature. input feature is 4-D, first 3 channels are all one, while the last channel is the degree of the vertex in the graph
        input_feature = graph_tree.treedict[0].x
        input_feature = input_feature[:, :3] 
        
        if global_config.gnn_input_signal == "all_one":
            input_feature = torch.ones_like(input_feature)  # use all-one input signal, as described in the paper
        elif global_config.gnn_input_signal == "xyz":
            pass     
        else:
            raise NotImplementedError("Unknown input signal type")
            
        # add a channel to represent the degree of the vertex
        degrees = torch_geometric.utils.degree(graph_tree.treedict[0].edge_index[0], num_nodes=graph_tree.treedict[0].x.shape[0], dtype=torch.float32)
        input_feature = torch.cat([input_feature, degrees.view(-1, 1)], dim=1)
        
        # Core network：forward the input mesh to the network, and get vertex-wise feature
        vert_feature = self.eigen_network(input_feature, graph_tree, graph_tree.depth)
      #  vert_feature = self.vert_decoder(vert_feature)
        
        torch.cuda.synchronize()

        ###########################################
        # STEP 2: Combine two vertexwise features to get edgewise feature
        ###########################################

        # get indexing of the two vertices of each edge
        vert_a_idx = graph_tree.treedict[0].edge_index[0]
        vert_b_idx = graph_tree.treedict[0].edge_index[1]
        vert_a_feature = vert_feature[vert_a_idx]
        # vert_a_feature = torch.index_select(vert_feature, vert_a_idx) 
        vert_b_feature = vert_feature[vert_b_idx]
        
        # predict the mass of each vertex
        if global_config.use_vertex_mass == True:
            vert_mass_predicted = self.mass_decoder(vert_feature)
        else:
            vert_mass_predicted = torch.ones([vert_feature.shape[0], 1], device=vert_feature.device)
        
        # predict the edge weight
        merged_feature = (vert_a_feature - vert_b_feature).pow(2)  # merge the two vertex features, and then do a MLP. note that the edge feature is symmetric
        edge_weight_predicted = self.edge_decoder(merged_feature)  
        
        if in_test:
            # during the test, we do not allow negative edge weight
            edge_weight_predicted = -torch.nn.functional.relu(-edge_weight_predicted)
        else:
            # during training, we allow negative edge weight. this is a bit helpful for training since it allow the network to explore more possibilities
            pass
        
        torch.cuda.synchronize()
 
        ###########################################
        # STEP 3: Compare the predicted edge result with the ground truth
        ###########################################
        
        # probe function

        if global_config.use_ref_mesh == True:
            vert_probe = self.probe_func.get_probe(graph_tree.real_underlying_graph, in_test=in_test, ref_graph_tree=graph_tree)
        else:
            vert_probe = self.probe_func.get_probe(graph_tree.treedict[0], in_test=in_test, ref_graph_tree=graph_tree)
            
        
        # vert probe * 10 to avoid small decimal numbers. Note that this has no effect in the resulting Laplacian --- Laplacian is a linear operator. feel free to remove this line
        vert_probe = vert_probe * 10
        
        # get the true edge weight
        if graph_tree.edge_weight is not None:
            true_edge_lap = graph_tree.edge_weight
            true_edge_lap = true_edge_lap.view(-1, 1)
        else:
            # if no true edge weight is provided, we will just set it to None
            console.log("It seems that you did not provide any true edge weight. We will set it to None.")
            true_edge_lap = None
        
        # vertex mass
        if global_config.use_vertex_mass == False:
            graph_tree.vertex_mass = torch.ones_like(vert_mass_predicted)
            vert_mass_predicted = torch.ones_like(vert_mass_predicted)
            
        # if real vertex mass is provided, we will use it. Otherwise, we will use all 1 as the mass
        if graph_tree.vertex_mass is not None:        
            vertex_mass = torch.clamp(graph_tree.vertex_mass, 0.1, 999) 
        else:
            console.log("It seems that you did not provide any true vertex mass. We will set it to all 1.")
            vertex_mass = torch.ones_like(vert_mass_predicted) * 1.0
        


        # Calculate the Laplacian operator after the real probe function
        after_lap_real = self.probe_func.lap_times_probe(graph_tree.real_underlying_graph, vert_probe, true_edge_lap, 
                                                        vertex_mass=vertex_mass, vertex_mass_inverse=True, automatic_add_self_loop=True,
                                                        ignore_error_and_return_anyway=True)   
        # Calculate the Laplacian operator after the predicted probe function
        after_lap_pred = self.probe_func.lap_times_probe(graph_tree.treedict[0], vert_probe, edge_weight_predicted, 
                                                        vertex_mass=vert_mass_predicted, vertex_mass_inverse=True, automatic_add_self_loop=True)
        
        
        # vertex-wise loss
        loss_dict = self.vertexwise_loss_func(  after_lap_pred, after_lap_real,
                                                smoothing_regulization_factor=global_config.smoothing_regulization_factor,
                                                edge_indices=graph_tree.treedict[0].edge_index,
                                                in_train=in_train,
                                                )
        # mass normalization loss
        vert_mass_predicted = vert_mass_predicted.view(-1, 1)
        vertex_mass = vertex_mass.view(-1, 1)
        loss_dict["mass_norm"] = ( (vert_mass_predicted - vertex_mass).abs().pow(2) ).mean() * 0.1    
        
        #loss_dict["mass_norm"] = ( (vert_mass_predicted - 1).mean().abs().pow(2) ) * 0.1
        loss_dict["loss_total"] += loss_dict["mass_norm"]
        
        # visualization: record the result of diffrenet laps
        loss_dict["after_lap_pred"] = after_lap_pred#torch.clamp(after_lap_pred, min=-0.1, max=0.1)
        loss_dict["after_lap_real"] = after_lap_real
        # log sparsity
        loss_dict["sparsity_ours"] = graph_tree.treedict[0].edge_index.shape[1] / (graph_tree.treedict[0].x.shape[0]) + 1 # +1 for self-loop
        
        
        
        ###########################################
        ###########################################
        
        # some legacy code of comparing with other methods
        loss_dict["loss_total_graph_lap_" + str(None)] = torch.zeros([1])
        loss_dict["loss_total_distance_based_lap_" + str(None)] = torch.zeros([1])
        loss_dict["loss_total_pseudo_heat_kernel_lap_" + str(None)] = torch.zeros([1])
        loss_dict["robust_L"] = 0    # [N, N]. ONLY contains the last one mesh!!!
        loss_dict["robust_M"] = 0    # [N].    ONLY contains the last one mesh!!!
        loss_dict["after_lap_robust"] = torch.zeros_like(after_lap_pred)
        loss_dict["after_lap_graph_lap"] = torch.zeros_like(after_lap_pred)
        loss_dict["after_lap_distance_based_lap"] = torch.zeros_like(after_lap_pred)
        loss_dict["after_lap_pseudo_heat_kernel_lap"] = torch.zeros_like(after_lap_pred)
        
        loss_dict["loss_total_robust"] = torch.zeros([1])
        loss_dict["loss_total_belkin"] = torch.zeros_like(loss_dict["loss_total_robust"])
        
        loss_dict["sparsity_graph_lap"] = 0
        loss_dict["sparsity_distance_based_lap"] = 0
        loss_dict["sparsity_pseudo_heat_kernel_lap"] = 0
        loss_dict["sparsity_robust"] = 0
        loss_dict["sparsity_belkin"] = 0
        
        loss_dict["per_probe_per_sample_belkin"] = torch.ones([112])
        loss_dict["per_probe_per_sample_robust"] = torch.ones([112])
        
       # if global_config.use_ref_mesh == False:
        #    compare_with_others = False
        
        if compare_with_others:
            import src.other_lbo_wrapper as lbo_wrapper
            graph_data = graph_tree.treedict[0]
            mass_naive = torch.ones_like(vert_mass_predicted, device=vert_mass_predicted.device)
            normalization = None
            
            # ordinary graph laplacian
            edge_idx, edge_weight = lbo_wrapper.get_graph_laplacian(graph=graph_data, normalization=normalization)
            new_graph = Data(x=graph_data.x, edge_index=edge_idx,)  # last line will add self-loop
            after_lap_result = self.probe_func.lap_times_probe(new_graph, vert_probe, edge_weight, 
                                                            vertex_mass=mass_naive, vertex_mass_inverse=True, automatic_add_self_loop=False) # 
            temp = self.vertexwise_loss_func(after_lap_result, after_lap_real)
            loss_dict["loss_total_graph_lap_" + str(normalization)] = temp["V_mse_loss"]
            loss_dict["per_probe_per_sample_graph_lap"] = temp["V_mse_loss_per_channel"]
            loss_dict["sparsity_graph_lap"] = new_graph.edge_index.shape[1] / (new_graph.x.shape[0])
            loss_dict["after_lap_graph_lap"] = after_lap_result
            
        
        # [Sharp and Crane 2020] 
        if compare_with_others and global_config.compare_with_robust_method == True:
            # edge weight
            after_lap_robust = []
            cum_sum = 0
            for i in range(self.batch_size):
                numpy_points = graph_tree.treedict[0][i].x.detach().cpu().numpy()[:, :3]
                cum_sum += numpy_points.shape[0]
                
                robust_L, robust_M = robust_laplacian.point_cloud_laplacian(numpy_points)
                # mass matrix
                from utils import relative_mass_matrix
            
                robust_M = relative_mass_matrix(robust_M, smallest_limit=0.1)    # clip the smallest value to 0.1, or it may numerically unstable
                
                # calculate the laplacian + probe
                temp_probe = vert_probe[cum_sum-numpy_points.shape[0]:cum_sum].cpu().numpy()
                temp_M_inv = np.diag(1 / robust_M.diagonal())
                temp_M_inv = coo_matrix(temp_M_inv)
                
                if global_config.use_vertex_mass:
                    temp_Lf = temp_M_inv @ robust_L @ temp_probe
                else:
                    raise NotImplementedError("not recommended!")
                    temp_Lf = robust_L @ temp_probe
                after_lap_robust.append(torch.tensor(temp_Lf, dtype=torch.float32).cuda())
            after_lap_robust = torch.cat(after_lap_robust, dim=0)
            robust_M = robust_M.diagonal()
            loss_dict["robust_L"] = robust_L    # [N, N]. ONLY contains the last one mesh!!!
            loss_dict["robust_M"] = robust_M    # [N].    ONLY contains the last one mesh!!!
            loss_dict["after_lap_robust"] = after_lap_robust
            temp = self.vertexwise_loss_func(after_lap_robust, after_lap_real)
            loss_dict["loss_total_robust"] = temp["V_mse_loss"]
            loss_dict["per_probe_per_sample_robust"] = temp["V_mse_loss_per_channel"]
            loss_dict["sparsity_robust"] = robust_L.nnz / robust_L.shape[0]
            

        else:
            loss_dict["after_lap_belkin"] = torch.ones_like(after_lap_real)

            
                    
            
        # probe
        loss_dict["original_color"] = vert_probe
        
        # mass
        loss_dict["pred_vert_mass"] = vert_mass_predicted
        loss_dict["true_vert_mass"] = vertex_mass
        #loss_dict["robust_vert_mass"] = robust_M   # ONLY contains the last one mesh!!!
        
        # edge attributes
        loss_dict["edge_weight_predicted"] = edge_weight_predicted
        loss_dict["true_edge_lap"] = true_edge_lap
        
        torch.cuda.synchronize()
        
        return loss_dict
        

    
    def training_step(self, batch, batch_idx):

        result = self.make_prediction(batch, batch_idx, compare_with_others=False)

        # get loss and log it
        loss_total = result["loss_total"]
        loss_mse = result["V_mse_loss"]
        loss_smoothing = result["smoothing_reg_loss"]
        loss_mass_norm = result["mass_norm"]
        mean_mass_after_norm = result["pred_vert_mass"].mean()
        
        self.log('overview/loss_total_train', loss_total, on_step=True, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
        self.log('overview/loss_mse_train', loss_mse, on_step=True, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
        self.log('overview/loss_smoothing_train', loss_smoothing, on_step=True, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
        
        self.log('overview/loss_mass_norm_train', loss_mass_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
        self.log('debug/mean_mass_train', mean_mass_after_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
        
        self.history_train_loss[self.current_epoch].append(loss_total.detach().cpu().numpy().mean())
    
        return loss_total
        
        
    
    def on_train_epoch_end(self) -> None:
        
        pass
    
    
    #############################################################
    
    def validation_step(self, batch, batch_idx):
        """
        The validation data should be taken as a reference only. The result may vary with different batch size.
        """
        
        if self.current_epoch == 0:
            pass
            return 
        
        should_visualize = (batch_idx == 0) and (((self.current_epoch+1) % global_config.visualize_every_n_epoch == 0) or (self.current_epoch == 0))
        
        result = self.make_prediction(batch, batch_idx, compare_with_others=True, in_test=True)
        
        loss_total = result["loss_total"]
        #V_rel_err = result["V_rel_err"]
        self.log('overview/loss_total_val', loss_total, on_step=False, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
        self.log('overview/loss_total_val_robust', result["loss_total_robust"], on_step=False, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
        #self.log('overview/V_rel_err_val', V_rel_err, on_step=False, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)

        t1 = 42
        t2 = t1+6
        t3 = t2+64
        assert result['original_color'].shape[1] == t3
        #assert result['after_lap_real'].shape[1] == t3
        #assert result['V_mse_loss_per_channel'].shape[0] == t3
        loss_A = result["V_mse_loss_per_channel"][0:t1].mean()
        loss_B = result["V_mse_loss_per_channel"][t1:t2].mean()
        loss_C = result["V_mse_loss_per_channel"][t2:t3].mean()
        self.log('val_per_probe_loss/Sin_Cos', loss_A, on_step=False, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
        self.log('val_per_probe_loss/XYZ', loss_B, on_step=False, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
        self.log('val_per_probe_loss/Eigen', loss_C, on_step=False, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
        
            
            
    def on_validation_epoch_end(self) -> None:
        
        pass


    #############################################################

    def test_step(self, batch, batch_idx):
        """
        test
        """
        
        assert global_config.batch_size == 1, "batch size must be 1 in test mode, so that we can better visualize the result. Or, this could avoid bias from per-channel balance"
        
        ############################################
        # test_visualize 
        ############################################
      #  print(batch['graph_tree'].mesh_name[0])
        if global_config.mode == "test_visualize":
            
            #console.log(f"Now we visualize the result of {batch['graph_tree'].mesh_name[0]}")
            h_graph = batch['graph_tree']
            result = self.make_prediction(batch, batch_idx, compare_with_others=True, in_test=True)
            loss_ours = result["loss_total"]
            loss_robus = result["loss_total_robust"]
            self.log('test_result/loss_total_test', loss_ours, on_step=False, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
            self.log('test_result/loss_total_test_robust', loss_robus, on_step=False, on_epoch=True, prog_bar=False, logger=True,  batch_size=self.batch_size, sync_dist=True)
            
            console.log("Now we visualize the result. This may take a while.")
            mesh_name = h_graph.mesh_name
            console.log(f"NOTE: You are now visualizing the mesh named: {mesh_name} (The path to the file may be inaccurate)" )
            
            ##############################################
                
            if global_config.do_visualizaion_on_plotly:
                # visualize via plotly.
                self.visualization_plotly(batch, result)
            
            ##############################################
            
        
        
    def on_test_epoch_end(self) -> None:
        pass
        
        

    ######################################################################
    # visualization
    ######################################################################        

    def visualization_plotly(self,
        batch, 
                             result, 
                             ):
        # visualize the our and [Sharp and Crane 2020] result for every specified channel via plotly
        to_visualize_channel = 95   # the 48th eigenvector
        vertices = batch['graph_tree'].treedict[0][0].x[:, :3].detach().cpu().numpy()       # [N, 3]
        pred = result["after_lap_pred"][:, to_visualize_channel].detach().cpu().numpy()     # [N,]
        robu = result["after_lap_robust"][:, to_visualize_channel].detach().cpu().numpy()   # [N,]
        real = result["after_lap_real"][:, to_visualize_channel].detach().cpu().numpy()     # [N,]
        
        from src.visualization.plotly_visualization import Visualizer
        # use same colorscale to compare
        
        visualizer_1 = Visualizer()
        visualizer_1.add_points(vertices, pred, cmin=real.min(), cmax=real.max())
        visualizer_1.show()
        
        visualizer_2 = Visualizer()
        visualizer_2.add_points(vertices, robu, cmin=real.min(), cmax=real.max())
        visualizer_2.show()
        
        visualizer_3 = Visualizer()
        visualizer_3.add_points(vertices, real, cmin=real.min(), cmax=real.max())
        visualizer_3.show()
        
        
    
        
    #################################################################
    
    


    ######################################################################
    # Optimizer
    ######################################################################
    
    def configure_optimizers(self):
        
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
        
        if self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError("optimizer not implemented")
        
        # linearly decay the learning rate to 0 when the training is finished
        if self.scheduler_name.lower() == "linear":
            scheduler = LinearLR(optimizer, start_factor=1., end_factor=0, total_iters=self.epochs)
        else:
            raise NotImplementedError("scheduler not implemented")
        
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
        
        return ret
    
    



