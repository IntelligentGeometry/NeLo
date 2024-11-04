import torch
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt

from config.global_config import global_config


def scatter_plot_2d(x, y, title=None, xlabel=None, ylabel=None, save_path="haha.png"):
    
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    if title != None:
        ax.set_title(title)
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    if save_path != None:
        fig.savefig(save_path)
    plt.close(fig)


def plot_distribution(x, title=None, xlabel=None, ylabel=None, save_path="haha.png"):
        
    fig, ax = plt.subplots()
    ax.hist(x, bins=200)
    if title != None:
        ax.set_title(title)
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    if save_path != None:
        fig.savefig(save_path)
    plt.close(fig)





def vertexwise_loss(
                    pred_vertex_attr,
                    real_vertex_attr,
                    return_per_channel_loss: bool=True,
                    
                    smoothing_regulization_factor: float=0.0,
                    edge_indices: torch.Tensor = None,
                    in_train: bool=False,
                     ):
    """
    calculate the vertex-wise loss
    return_per_channel_loss: whether to return the loss of each channel. Generally used for visualization debug.
    smoothing_regulization_factor: deprecated
    edge_indices: [2, N]. deprecated
    in_train: if in train, there will be some tricks to maintain numerical stability of training. If False, the tricks will not take effect, and all Loss will be clipped to within 1.0, as described in the paper.
    
    returns:
        - loss_total: scalar
        - V_mse_loss: tensor, shape: (n_channels,) 
    """
    
    loss_multiplier = 1.0
     
    # if we are training, we will calculate a weight factor for each channel, so that we can "average out" all channels during training to avoid one channel being too dominant
    if global_config.process_before_calculate_loss == "factor" and in_train==True:
        per_channel_mean = real_vertex_attr.abs().mean(dim=0)   # shape: (n_channels,)
        loss_factor = 1.0 / (per_channel_mean + 0.1)           # shape: (n_channels,)
        #loss_factor = torch.log10(loss_factor + 1.0)               # shape: (n_channels,)
    else:
        loss_factor = 1.0
        
   # print(per_channel_mean)
   # print("===")
   # print(loss_factor)
    
    # get per-channel mse loss
    mse_loss = torch.nn.MSELoss(reduction="none")(pred_vertex_attr, real_vertex_attr)
    # shape: (n_vertices, n_channels)
    mse_loss = mse_loss.mean(dim=0)                             # shape: (n_channels,)
    mse_loss = (mse_loss * loss_factor)                         # shape: (n_channels,)
    # if not in training, we will clamp the loss to within 1.0, as described in the paper
    if in_train == False:
        pass
#        mse_loss = torch.clamp(mse_loss, -999, 1)
        
    mse_loss_final = mse_loss.mean()                   # shape: scalar
    
    relative_error = torch.abs(pred_vertex_attr - real_vertex_attr) / (torch.abs(real_vertex_attr) + 0.001)
    mean_relative_error = relative_error.mean()
    
    # add smoothing regulization
    if smoothing_regulization_factor != 0.0:
        vert_a_attr = pred_vertex_attr[edge_indices[0]]     # shape: (n_edges, n_channels)
        vert_b_attr = pred_vertex_attr[edge_indices[1]]     # shape: (n_edges, n_channels)
        # compute the difference between the two vertices
        smoothing_reg_loss = vert_a_attr - vert_b_attr      # shape: (n_edges, n_channels)
        smoothing_reg_loss = smoothing_reg_loss.abs().mean()    # float
        smoothing_reg_loss *= smoothing_regulization_factor     # float
    else:
        smoothing_reg_loss = 0.0
    
    
    # return results
    results = {
        "loss_total": mse_loss_final + smoothing_reg_loss,      # scalar
        "V_mse_loss": mse_loss_final,                           # scalar
        "V_rel_err": mean_relative_error,                       # scalar
        "smoothing_reg_loss": smoothing_reg_loss,               # scalar
    }
    if return_per_channel_loss:         
        # return the loss of each channel. Generally used for debug.
        results["V_mse_loss_per_channel"] = mse_loss            # shape: (n_channels,)
        
  
    
    # NAN check
    if torch.isnan(results["loss_total"]) or torch.isinf(results["loss_total"]):
        print("NAN encountered!!!")
    #    breakpoint()
    
    # training monitor
    threshold = 1
    if results["loss_total"] > threshold and in_train:
        print(f"The loss > {threshold}, This may suggest an error.")
        #results["loss_total"] = torch.clamp(results["loss_total"], -999, threshold)
    
    return results
    

