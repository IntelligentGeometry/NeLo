# this file provide the dataloader for the dataset defined in ./data/eigens

from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import multiprocessing
from torch._C import device

#from torch_geometric.data import Data
import torch_geometric
import torch.utils.data
import numpy as np
import os
from tqdm import tqdm
from rich.progress import Progress
import trimesh
import pickle
from joblib import Parallel, delayed
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from scipy.sparse import coo_matrix
import scipy.sparse as sp

#from torch_geometric.loader import DataLoader #as PygDataLoader

from torch.utils.data import Dataset

from src.graphtree.graph_tree import GraphTree, construct_graph_tree_from_mesh, construct_graph_tree_from_point_cloud

from config.global_config import global_config, console

import utils



def determine_cache_folder():
    pass


class LapDataset(Dataset):
    """
    a dataset class for the laplacian dataset
    """
    def __init__(self, graph_trees: list[GraphTree] = None, graph_trees_names: list[str] = None):
        
        super().__init__()

        #for graph_tree in graph_trees:
        #    graph_tree.index_correction()
        self.graph_trees = graph_trees
        self.graph_trees_names = graph_trees_names
        #self.graph_trees_backup = graph_trees

    def __len__(self):
        # return the number of meshes in the dataset
        if global_config.preload_all_data_into_memory == True:
            return len(self.graph_trees)
        else:
            return len(self.graph_trees_names)

    
    def __getitem__(self, idx):
        # return the idx-th mesh in the dataset
        # idx: int
        # return: dict
        
        if global_config.preload_all_data_into_memory == True:
            # if the data is already loaded into memory, just return it
            graph_tree = self.graph_trees[idx]
            graph_tree.index_correction()
            return {
                "graph_tree": graph_tree,
            }

        else:
            # load from the disk
            name = self.graph_trees_names[idx]
            
            # 
            if "train" in self.graph_trees_names[idx]:
                is_train = True
            else:
                is_train = False
                
            if is_train:
                if global_config.using_wang_2016_kinect_dataset == True:
                    gauss_noise_strength = "kinect"
                else:
                    gauss_noise_strength = global_config.random_move_vertex_strength_train
            else:
                if global_config.using_wang_2016_kinect_dataset == True:
                    gauss_noise_strength = "kinect"
                else:
                    gauss_noise_strength = global_config.random_move_vertex_strength_val

                
            # check if it has already been cached
            cache_path = name.replace("_meshes", "_cache_" + str(gauss_noise_strength))
            cache_path = cache_path.replace(".obj", ".pkl")
            if os.path.exists(cache_path):
                # already cached, load it and return
                with open(cache_path, "rb") as f:
                    graph_tree = pickle.load(f)
                return  {
                    "graph_tree": graph_tree,
                }
            else:
                raise RuntimeError("You have to cache first!")
 
            



#@profile
def my_collate_func(batch: list[dict]) -> dict:
    '''
    My implementation of batch collection
    see https://plainenglish.io/blog/understanding-collate-fn-in-pytorch-f9d1742647d3 for tutorial
    '''
    outputs = {}
    
    # iterate all keys in the batch
    for each in batch[0].keys():
       # for i in range(batch.__len__()):
        #    print(i, batch[i]['graph_tree'].treedict[1].edge_index.min(), batch[i]['graph_tree'].treedict[1].edge_index.max())
        # we merge graph_tree
        if each == "graph_tree":
            graphtree_super = GraphTree(batch_size=len(batch))
            
            if global_config.num_workers == 0 and False:
                graph_tree_list = [b['graph_tree'].cuda() for b in batch]
            else:
                graph_tree_list = [b['graph_tree'] for b in batch]
            # TODO: merge graph tree may need some optimization to speed up
            graphtree_super.merge_graphtree(graph_tree_list)
            outputs['graph_tree'] = graphtree_super#.cuda()
        # merge real_eigenvectors / real_eigenvalues 
        else:
            pass
    
    # return the new dict we just constructed
    return outputs


class MyLapDataset(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        # load the config
        self.data_path = global_config.data_folder
        self.batch_size = global_config.batch_size
        self.load_ratio = global_config.load_ratio
        self.num_workers = global_config.num_workers  
        self.train_graph_trees = []
        self.val_graph_trees = []
        self.test_graph_trees = []
        
        self.names_train = []                 # the names of the train meshes, for debugging
        self.names_val = []                   # the names of the val meshes, for debugging


    def construct_and_precache_h_graph(self,
                                            mesh_path: str, 
                                            is_train: bool = True,
                                            ) -> None:
        """
        
        """
        
        if is_train:
            if global_config.using_wang_2016_kinect_dataset == True:
                gauss_noise_strength = "kinect"
            else:
                gauss_noise_strength = global_config.random_move_vertex_strength_train
        else:
            if global_config.using_wang_2016_kinect_dataset == True:
                gauss_noise_strength = "kinect"
            else:
                gauss_noise_strength = global_config.random_move_vertex_strength_val
                
        if gauss_noise_strength is None:
            gauss_noise_strength = 0.0
            
        # first, check if it has already been cached
        cache_path = mesh_path.replace("_meshes", "_cache_" + str(gauss_noise_strength))
        #if global_config.point_cloud_knn_k != 8:
        #    cache_path += "_knn_" + str(global_config.point_cloud_knn_k)
        cache_path = cache_path.replace(".obj", ".pkl")
        if os.path.exists(cache_path):
            return
        
        # if not cached, load the mesh
        mesh = trimesh.load(mesh_path, process=False)
        vertices = mesh.vertices
        faces = mesh.faces
        
        # for wang 2016 kinect dataset, we need to load the noisy mesh
        if global_config.using_wang_2016_kinect_dataset == True:
            # noisy verts
            noisy_mesh_path = mesh_path.replace("_meshes", "_noisy_meshes")
            noisy_mesh_path = noisy_mesh_path.replace(".obj", "_noisy.obj")
            noisy_mesh = trimesh.load(noisy_mesh_path, process=False)
            vertices = noisy_mesh.vertices  # use noisy vertices
            
        elif gauss_noise_strength > 0.0:
            # add noise to the vertices, if needed
            vertices = utils.add_noise_to_vertices_position(vertices, strength=gauss_noise_strength)
            
            
        # generate graph tree!
        if global_config.train_graph_data_construt == "mesh":
            graph_tree = construct_graph_tree_from_mesh(vertices, faces)
            assert global_config.use_data_augmentation_train == False 
        elif global_config.train_graph_data_construt == "pc":
            graph_tree = construct_graph_tree_from_point_cloud(vertices, ref_mesh=mesh, mesh_name=mesh_path)
        else:
            raise NotImplementedError
        
        # cache the graph tree
        with open(cache_path, "wb") as f:
            pickle.dump(graph_tree, f)
        return graph_tree
    
    
    def load_cached_h_graph(self, mesh_path: str, is_train:bool =True) -> GraphTree:
        """
        
        """        
        
        if is_train:
            if global_config.using_wang_2016_kinect_dataset == True:
                gauss_noise_strength = "kinect"
            else:
                gauss_noise_strength = global_config.random_move_vertex_strength_train
        else:
            if global_config.using_wang_2016_kinect_dataset == True:
                gauss_noise_strength = "kinect"
            else:
                gauss_noise_strength = global_config.random_move_vertex_strength_val
                
        if gauss_noise_strength is None:
            gauss_noise_strength = 0.0
       # breakpoint()
            
        # first, check if it has already been cached
        cache_path = mesh_path.replace("_meshes", "_cache_" + str(gauss_noise_strength))
        #if global_config.point_cloud_knn_k != 8:
         #   cache_path += "_knn_" + str(global_config.point_cloud_knn_k)
        cache_path = cache_path.replace(".obj", ".pkl")
        if os.path.exists(cache_path):
            # already cached, load it and return
            with open(cache_path, "rb") as f:
                graph_tree = pickle.load(f)
         #   if graph_tree.layers_eigen_vecs.keys().__len__() == 0:
          #      raise RuntimeError("The graph tree loaded from cache has no eigen vecs!")
            return graph_tree
        else:
            # not cached. Let's just construct it and cache it
            self.construct_and_precache_h_graph(mesh_path, is_train=is_train)
            with open(cache_path, "rb") as f:
                graph_tree = pickle.load(f)
            return graph_tree
    
    
    
    def cache_pkl(self):
        """
        For all the meshes in the dataset, we cache the corresponding h_graphs.
        """
        
        # mkdir of the cache folders
        if global_config.using_wang_2016_kinect_dataset == True:
            gauss_noise_strength_train = "kinect"
        else:
            gauss_noise_strength_train = global_config.random_move_vertex_strength_train

        if global_config.using_wang_2016_kinect_dataset == True:
            gauss_noise_strength_val = "kinect"
        else:
            gauss_noise_strength_val = global_config.random_move_vertex_strength_val


        if not os.path.exists(self.data_path + "/train_cache_" + str(gauss_noise_strength_train)):
            os.mkdir(self.data_path + "/train_cache_" + str(gauss_noise_strength_train))
        if not os.path.exists(self.data_path + "/val_cache_" + str(gauss_noise_strength_val)):
            os.mkdir(self.data_path + "/val_cache_" + str(gauss_noise_strength_val))
        if not os.path.exists(self.data_path + "/test_cache_" + str(gauss_noise_strength_val)):
            os.mkdir(self.data_path + "/test_cache_" + str(gauss_noise_strength_val))


        # load the training meshes
        self.names_train = os.listdir(self.data_path + "/train_meshes")
        self.names_train = [name for name in self.names_train if name.endswith(".obj")]
        Parallel(n_jobs=global_config.load_data_workers)(
            delayed(self.load_cached_h_graph)(os.path.join(self.data_path + "/train_meshes", name), is_train=True) for name in tqdm(self.names_train)
        )
        console.log("Finish loading training data!", style="red")
        
        # load val meshes
        self.names_val = os.listdir(self.data_path + "/val_meshes")
        self.names_val = [name for name in self.names_val if name.endswith(".obj")]
        Parallel(n_jobs=global_config.load_data_workers)(
            delayed(self.load_cached_h_graph)(os.path.join(self.data_path + "/val_meshes", name), is_train=False) for name in tqdm(self.names_val)
        )
        console.log("Finish loading val data!", style="red")
        
        # load test meshes
        self.names_test = os.listdir(self.data_path + "/test_meshes")
        self.names_test = [name for name in self.names_test if name.endswith(".obj")]
        Parallel(n_jobs=global_config.load_data_workers)(
            delayed(self.load_cached_h_graph)(os.path.join(self.data_path + "/test_meshes", name), is_train=False) for name in tqdm(self.names_test)
        )
        console.log("Finish loading test data!", style="red")
        console.log("Load training data done!", style="red")
    
        
    def prepare_data(self):
        pass
        


    def setup(self, stage=None):
        
        self.persistent_workers = True if global_config.num_workers > 0 else False
        # Note: pin-memory is only available when no sparse tensors are involved
        # persistent_workers is only available when num_workers > 0
        
        # mkdir of the cache folders
        gauss_noise_strength_train = global_config.random_move_vertex_strength_train
        gauss_noise_strength_val = global_config.random_move_vertex_strength_val
        if not os.path.exists(self.data_path + "/train_cache_" + str(gauss_noise_strength_train)):
            os.mkdir(self.data_path + "/train_cache_" + str(gauss_noise_strength_train))
        if not os.path.exists(self.data_path + "/val_cache_" + str(gauss_noise_strength_val)):
            os.mkdir(self.data_path + "/val_cache_" + str(gauss_noise_strength_val))
        if not os.path.exists(self.data_path + "/test_cache_" + str(gauss_noise_strength_val)):
            os.mkdir(self.data_path + "/test_cache_" + str(gauss_noise_strength_val))
        
        # get all the names of the training meshes
        names_train = os.listdir(self.data_path + "/train_meshes")
        self.names_train = []
        for name in names_train:
            if name.endswith(".obj"):       # add to the list only if it is a .obj file
                self.names_train.append(name)
                
        # get all the names of the validation meshes
        names_val = os.listdir(self.data_path + "/val_meshes")
        self.names_val = []
        for name in names_val:
            if name.endswith(".obj"):    # add to the list only if it is a .obj file
                self.names_val.append(name)
        
        # get all the names of the test meshes
        names_test = os.listdir(self.data_path + "/test_meshes")
        self.names_test = []
        for name in names_test:
            if name.endswith(".obj"):   # add to the list only if it is a .obj file
                self.names_test.append(name)
        
        ################################################
        ################################################
        if global_config.preload_all_data_into_memory == True: 
            # load all the data into memory will be much faster, but will consume more memory
                    
            # put them into memory
            for name in tqdm(self.names_train):
                if self.load_ratio < 1.0:
                    if np.random.rand() > self.load_ratio:
                        continue
                self.train_graph_trees.append(self.load_cached_h_graph(os.path.join(self.data_path + "/train_meshes", name), is_train=True))
            for name in tqdm(self.names_val):
                self.val_graph_trees.append(self.load_cached_h_graph(os.path.join(self.data_path + "/val_meshes", name), is_train=False))
            for name in tqdm(self.names_test):
                self.test_graph_trees.append(self.load_cached_h_graph(os.path.join(self.data_path + "/test_meshes", name), is_train=False))
            
            # get the index of the current gpu
            current_gpu_idx = torch.cuda.current_device()
            # only print the info on the first gpu
            if current_gpu_idx == 0:
                console.log("Finish loading data!", style="red")
                console.log("Load training data done, total:", len(self.train_graph_trees), "objects with avg #verts:", str(np.array([len(v.treedict[0].x) for v in self.train_graph_trees]).mean()), style="red")
                console.log("Load validation data done, total:", len(self.val_graph_trees), "objects with avg #verts:", str(np.array([len(v.treedict[0].x) for v in self.val_graph_trees]).mean()), style="red")
                console.log("Load test data done, total:", len(self.test_graph_trees), "objects with avg #verts:", str(np.array([len(v.treedict[0].x) for v in self.test_graph_trees]).mean()), style="red")
                
            # build the dataset
            self.train_dataset = LapDataset(self.train_graph_trees
                                            )
            self.val_dataset = LapDataset(self.val_graph_trees
                                            )
            self.test_dataset = LapDataset(self.test_graph_trees
                                            )
            
            
        else:
            # lazy loading the data
            
            # if the load_ratio is not 1.0, we need to sample a subset of the data
            if self.load_ratio < 1.0:
                if self.load_ratio < 0.0:
                    raise RuntimeError("load_ratio should be in [0.0, 1.0]")
                self.names_train = self.names_train[:int(len(self.names_train) * self.load_ratio)]
                self.names_val = self.names_val[:int(len(self.names_val) * self.load_ratio)]
                self.names_test = self.names_test[:int(len(self.names_test) * self.load_ratio)]
            elif self.load_ratio > 1.0:
                self.load_ratio = 1.0
            #
            # get the index of the current gpu
            current_gpu_idx = torch.cuda.current_device()
            # only print the info on the first gpu
            if current_gpu_idx == 0:
                console.log(f"Lazy loading the dataset... in lazy mode, please make sure that the dataset pkl has already been cached! (load_ratio: {self.load_ratio})", style="red")
                console.log("Finding training data done, total:", len(self.names_train), style="red")
                console.log("Finding validation data done, total:", len(self.names_val), style="red")
                console.log("Finding test data done, total:", len(self.names_test), style="red")
            
            train_names = [os.path.join(self.data_path + "/train_meshes", name) for name in self.names_train]
            val_names = [os.path.join(self.data_path + "/val_meshes", name) for name in self.names_val]
            test_names = [os.path.join(self.data_path + "/test_meshes", name) for name in self.names_test]
            
            # build the dataset
            self.train_dataset = LapDataset(graph_trees_names=train_names
                                            )
            self.val_dataset = LapDataset(graph_trees_names=val_names
                                            )
            self.test_dataset = LapDataset(graph_trees_names=test_names
                                            )
            
    
    def train_dataloader(self):
        ctx = multiprocessing.get_context("spawn")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=my_collate_func, persistent_workers=self.persistent_workers, pin_memory=False)  # , prefetch_factor=6

    def val_dataloader(self):
        ctx = multiprocessing.get_context("spawn")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=global_config.shuffle_val_dataset, persistent_workers=self.persistent_workers, collate_fn=my_collate_func, pin_memory=False)

    def test_dataloader(self):
        ctx = multiprocessing.get_context("spawn")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=global_config.shuffle_val_dataset, collate_fn=my_collate_func, persistent_workers=self.persistent_workers, pin_memory=False)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError


   # def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
   #     print("ssdfsdfsdf")
   #     return super().transfer_batch_to_device(batch, device, dataloader_idx)



