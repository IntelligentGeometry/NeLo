import os
import trimesh
import numpy as np
import time
import torch
from src.graphtree.data import Data
import matplotlib

import matplotlib.pyplot as plt

from visualization.color_mapping import scalar_color_mapping



MESH_PATH = "./out/mesh"
if not os.path.exists(MESH_PATH):
    os.makedirs(MESH_PATH)
    
#############################################
# Utils
#############################################

def single_channel_color_normalize(colors, mode="auto"):
    """
    colors: [N, 1] or [N]
    mode:
        None/Fasle: no normalization
        True/"auto": if the range of colors is too small, return a constant color, else normalize the colors to [0, 1]
    """
    if mode is None or mode is False:
        return colors
    elif mode == "auto" or mode is True:
        if colors.max() - colors.min() < 0.001:
            return np.ones_like(colors) * 0.5
        else:
            return (colors - colors.min()) / (colors.max() - colors.min())
            
    else:
        raise NotImplementedError("Unknown mode: {}".format(mode))


