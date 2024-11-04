"""
refer to https://matplotlib.org/stable/users/explain/colors/colormaps.html for other color maps
"""

import numpy as np
from collections import defaultdict
import colorcet as cc


                      
                      
                      
def using_colorcet_color_map(colors: np.ndarray,
                                color_map: str="CET_R3",  ):
    """
    https://colorcet.holoviz.org/user_guide/Continuous.html
    """
    if colors.ndim == 1:
        colors = colors.reshape(-1, 1)
    assert colors.ndim == 2
    
    # get 256 colors from colorcet
    if color_map == "CET_R3":               
        # pink to yellow to blue, used for probes
        cc_256_colors = cc.CET_R3[:256]   
    elif color_map == "coolwarm" or color_map == "CET_D1":
        # blue to white to red,
        cc_256_colors = cc.coolwarm[:256]
    elif color_map == "CET_D1A":
        # blue to white to red, but darker than coolwarm
        cc_256_colors = cc.CET_D1A[:256]    
    elif color_map == "bmy":
        # blue to magenta to yellow
        cc_256_colors = cc.bmy[:256]        
        
    else:
        raise ValueError("Unknown color map!")
    
    # now cc_256_colors are something like [#dd1422', '#dc121e', ...], we need to convert them to RGB
    cc_256_colors = np.array([list(int(cc_256_colors[i][1:3], 16) for i in range(256)),
                              list(int(cc_256_colors[i][3:5], 16) for i in range(256)),
                              list(int(cc_256_colors[i][5:7], 16) for i in range(256))]).T
                             
    # 0-1 map to 0-255
    colors = (colors * 255).astype(np.int32)
    
   # print(colors, colors.shape)
    #breakpoint()
    #print(cc_256_colors)

    R = cc_256_colors[colors, 0]
    G = cc_256_colors[colors, 1]
    B = cc_256_colors[colors, 2]
    R, G, B = R / 255, G / 255, B / 255
    return np.concatenate([R, G, B], axis=1)


def scalar_color_mapping(colors: np.ndarray,
                         color_map: str,
                         test_all: bool = False
                         ):
    """
    colors: [N, 1] or [N]
    color_map: str
    test_all: bool, if True, will test all color maps and save the results to files
    return: [N, 3]
    """ 
    
    blue_cyan_yellow_red = [[0,55,77], [25,254,245], [253,238,218], [253,0,64], [173,37,44]]
    
    all_color_maps = dict()
    
    all_color_maps['green_to_red'] = scalar_color_mapping_green_red
    all_color_maps['blue_to_red'] = scalar_color_mapping_blue_red
    all_color_maps['blue_to_red_dimmed'] = lambda x: scalar_color_mapping_blue_red(x, dimmed=True)
    all_color_maps['coolwarm'] = lambda x: using_colorcet_color_map(x, color_map='coolwarm')
    all_color_maps['CET_D1A'] = lambda x: using_colorcet_color_map(x, color_map='CET_D1A')
    all_color_maps['blue_cyan_yellow_red'] = using_colorcet_color_map
    all_color_maps['rainbow'] = scalar_color_mapping_rainbow
    all_color_maps['white_to_red'] = scalar_color_mapping_white_red
    all_color_maps['white_to_pink'] = scalar_color_mapping_white_to_pink
    all_color_maps['pure_white'] = pure_white_color
    all_color_maps['purple_to_green'] = scalar_color_mapping_purple_to_green
    all_color_maps['purple_to_green_to_red'] = purple_to_green_to_red
    
    ########################
    # only when test_all is True
    if test_all and __name__ != "__main__":
        raise ValueError("test_all can only be used when this file is executed alone!")
    elif test_all:
        for color_map in all_color_maps.keys():
            import matplotlib.pyplot as plt
            colors = np.linspace(0, 1, 100).reshape(-1, 1)
            colors_rgb = scalar_color_mapping(colors, color_map)
            plt.figure()
            plt.scatter(colors, np.zeros_like(colors), c=colors_rgb)
            plt.title(color_map)
            plt.savefig(f"{color_map}.png")
            print(f"Saved {color_map}.png")
    # 
    ########################
        
    if color_map in all_color_maps.keys():
        return all_color_maps[color_map](colors)
    else:
        raise ValueError("Unknown color map!")

##########################################################
# 
##########################################################

def scalar_color_mapping_green_red(colors):
    """
    colors: [N, 1] or [N]
    """
    if colors.ndim == 1:
        colors = colors.reshape(-1, 1)
    assert colors.ndim == 2
    # to RGB
    R = np.clip(1 * colors, 0, 1)
    G = np.clip(1 - 1 * colors, 0, 1)
    B = np.zeros_like(colors)
    return np.concatenate([R, G, B], axis=1)


def scalar_color_mapping_blue_red(colors, dimmed=False):
    """
    colors: [N, 1] or [N]
    dimmed: bool
    """
    if colors.ndim == 1:
        colors = colors.reshape(-1, 1)
    assert colors.ndim == 2
    
    R, G, B = np.zeros_like(colors), np.zeros_like(colors), np.zeros_like(colors)

    for i in range(colors.shape[0]):    
        if colors[i] < 0.25:
            R[i] = 0
            G[i] = 4 * colors[i]
            B[i] = 1
        elif colors[i] < 0.5:
            R[i] = 0
            G[i] = 1
            B[i] = 1 - 4 * (colors[i] - 0.25)
        elif colors[i] < 0.75:
            R[i] = 4 * (colors[i] - 0.5)
            G[i] = 1
            B[i] = 0
        else:
            R[i] = 1
            G[i] = 1 - 4 * (colors[i] - 0.75)
            B[i] = 0
            
    if dimmed:
        R = R * 0.6
        G = G * 0.6
        B = B * 0.6
            
    return np.concatenate([R, G, B], axis=1)



    
    
    
    
def scalar_color_mapping_rainbow(colors):
    """
    colors: [N, 1] or [N]
    """
    if colors.ndim == 1:
        colors = colors.reshape(-1, 1)
    assert colors.ndim == 2
    
    R, G, B = np.zeros_like(colors), np.zeros_like(colors), np.zeros_like(colors)
    for i in range(colors.shape[0]):
        if colors[i] < 0.2:
            R[i] = 1
            G[i] = 5 * colors[i]
            B[i] = 0
        elif colors[i] < 0.4:
            R[i] = 1 - 5 * (colors[i] - 0.2)
            G[i] = 1
            B[i] = 0
        elif colors[i] < 0.6:
            R[i] = 0
            G[i] = 1
            B[i] = 5 * (colors[i] - 0.4)
        elif colors[i] < 0.8:
            R[i] = 0
            G[i] = 1 - 5 * (colors[i] - 0.6)
            B[i] = 1
        else:
            R[i] = 5 * (colors[i] - 0.8)
            G[i] = 0
            B[i] = 1
        
    return np.concatenate([R, G, B], axis=1)
    
    

def scalar_color_mapping_white_red(colors):
    """
    colors: [N, 1] or [N]
    """
    if colors.ndim == 1:
        colors = colors.reshape(-1, 1)
    assert colors.ndim == 2
    # # to RGB
    R = np.ones_like(colors)
    G = np.clip(1 - 1 * colors, 0, 1)
    B = np.clip(1 - 1 * colors, 0, 1)
    return np.concatenate([R, G, B], axis=1)


def scalar_color_mapping_white_to_pink(colors):
    """
    colors: [N, 1] or [N]
    """
    if colors.ndim == 1:
        colors = colors.reshape(-1, 1)
    assert colors.ndim == 2
    # linear mapping between #f768a1 and #ffffff
    R = np.clip(247 + 9 * (1-colors), 0, 255)
    G = np.clip(104 + 151 * (1-colors), 0, 255)
    B = np.clip(161 + 94 * (1-colors), 0, 255)
    R, G, B = R / 255, G / 255, B / 255
    return np.concatenate([R, G, B], axis=1)
    
    
def pure_white_color(colors):
    """
    colors: [N, 1] or [N]
    """
    if colors.ndim == 1:
        colors = colors.reshape(-1, 1)
    assert colors.ndim == 2
    # # to RGB
    R = np.ones_like(colors)
    G = np.ones_like(colors)
    B = np.ones_like(colors)
    return np.concatenate([R, G, B], axis=1)



def three_color_mapping(colors,
                        zero_percent_color,
                        fifty_percent_color,
                        hundred_percent_color):
    """
    """
    if colors.ndim == 1:
        colors = colors.reshape(-1, 1)
    assert colors.ndim == 2
    # for color < 0.5, linear mapping between zero_percent_color and fifty_percent_color
    # for color >= 0.5, linear mapping between fifty_percent_color and hundred_percent_color
    R = np.zeros_like(colors)
    G = np.zeros_like(colors)
    B = np.zeros_like(colors)
    for i in range(colors.shape[0]):
        if colors[i] < 0.5:
            R[i] = zero_percent_color[0] + (fifty_percent_color[0] - zero_percent_color[0]) * 2 * colors[i]
            G[i] = zero_percent_color[1] + (fifty_percent_color[1] - zero_percent_color[1]) * 2 * colors[i]
            B[i] = zero_percent_color[2] + (fifty_percent_color[2] - zero_percent_color[2]) * 2 * colors[i]
        else:
            R[i] = fifty_percent_color[0] + (hundred_percent_color[0] - fifty_percent_color[0]) * 2 * (colors[i] - 0.5)
            G[i] = fifty_percent_color[1] + (hundred_percent_color[1] - fifty_percent_color[1]) * 2 * (colors[i] - 0.5)
            B[i] = fifty_percent_color[2] + (hundred_percent_color[2] - fifty_percent_color[2]) * 2 * (colors[i] - 0.5)
    R, G, B = R / 255, G / 255, B / 255
    return np.concatenate([R, G, B], axis=1)



def scalar_color_mapping_purple_to_green(colors):
    """
    colors: [N, 1] or [N]
    """
    zero_percent_color = np.array([133, 26, 83])
    fifty_percent_color = np.array([255, 255, 255])
    hundred_percent_color = np.array([56, 99, 37])
    return three_color_mapping(colors, zero_percent_color, fifty_percent_color, hundred_percent_color)
    
def purple_to_green_to_red(colors):
    zero_percent_color = np.array([114, 21, 244])
    fifty_percent_color = np.array([170, 250, 170])
    hundred_percent_color = np.array([234, 52, 36])
    return three_color_mapping(colors, zero_percent_color, fifty_percent_color, hundred_percent_color)

##########################################################


if __name__ == "__main__":
    scalar_color_mapping(None, None, test_all=True)
