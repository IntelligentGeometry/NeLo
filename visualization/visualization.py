import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import igl

import robust_laplacian
import scipy.sparse.linalg as sla

import sys
sys.path.append(".")
import utils
sys.path.append("./visualization")
from littlerender_binding import log_mesh


def visualize_two_vectors(vector_1, vector_2, normalize=False, output_path="out/visualization.png"):
    # this function visualize two vector in one figure
    # vector_1: [dim]
    # vector_2: [dim]
    # return: None
    
    # check whether the two vectors have the same dimension
    assert vector_1.shape == vector_2.shape
    # convert the vectors to numpy array, if they are torch tensors
    if isinstance(vector_1, torch.Tensor):
        vector_1 = vector_1.detach().cpu().numpy()
    if isinstance(vector_2, torch.Tensor):
        vector_2 = vector_2.detach().cpu().numpy()
    
    # normalize the vector
    if normalize == True:
        vector_1 = vector_1 / np.linalg.norm(vector_1)
        vector_2 = vector_2 / np.linalg.norm(vector_2)
        
    # plot the figure
    figure = plt.figure()
    # set the resolution of the figure
    figure.set_dpi(500)
    # plot the two vectors
    subfigure_1 = figure.add_subplot(2, 1, 1)
    subfigure_2 = figure.add_subplot(2, 1, 2)
    subfigure_1.bar(range(len(vector_1)), vector_1)
    subfigure_2.bar(range(len(vector_2)), vector_2)
    subfigure_1.plot(range(len(vector_1)), np.zeros_like(vector_1), color="red")
    subfigure_2.plot(range(len(vector_2)), np.zeros_like(vector_2), color="red")
    # add the x-axis
    subfigure_1.set_xlabel("index")
    subfigure_2.set_xlabel("index")
    
    # save the figure
    plt.savefig(output_path)
    plt.close()
    
    
    

def visualize_an_array(array, output_path="out/visualization_matrix.png"):
    # given a 2D array (matrix), visualize it
    # array: [N, M]
    
    figure = plt.figure()
    figure.set_dpi(500)
    figure.add_subplot(1, 1, 1)
    plt.imshow(array)
    plt.savefig(output_path)
    plt.close()


    
    