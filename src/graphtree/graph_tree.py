from __future__ import annotations
from typing import Callable, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import SAGEConv
import numpy as np
#from torch_geometric.nn import avg_pool, voxel_grid

#from utils.thsolver import default_settings

#from src.graphtree.decide_edge_type import *
from src.graphtree.data import Data, convert_mesh_to_graph_data, convert_point_cloud_to_graph

from src.modules.poolings import PoolingGraph
from src.modules.poolings import UnpoolingGraph

import visualization.littlerender_binding as littlerender_binding


from scipy.sparse import coo_matrix

import scipy.sparse as sp
import scipy.sparse.linalg as sla

import trimesh
from utils import get_GT_L_and_M, get_GT_eigens_of_mesh, convert_dense_tensor_to_sparse_numpy, convert_sparse_numpy_to_dense_tensor, relative_mass_matrix

from config.global_config import global_config





def avg_pool(
    cluster: torch.Tensor,
    data: Data,
    transform: Optional[Callable] = None,
) -> Data:
    """a wrapper of torch_geometric.nn.avg_pool"""
    data_torch_geometric = torch_geometric.data.Data(x=data.x, edge_index=data.edge_index.contiguous())
    new_data = torch_geometric.nn.avg_pool(cluster, data_torch_geometric, transform=transform)
    ret = Data(x=new_data.x, edge_index=new_data.edge_index)
    return ret


def avg_pool_maintain_old(
    cluster: torch.Tensor,
    data: Data,
    transform: Optional[Callable] = None,
):
    """a wrapper of torch_geometric.nn.avg_pool, but maintain the old graph"""
    data_torch_geometric = torch_geometric.data.Data(x=data.x, edge_index=data.edge_index)
    new_layer = torch_geometric.nn.avg_pool(cluster, data_torch_geometric, transform=transform)
    # connect the corresponding node in the two layers
    


def pooling(data: Data, size: float, normal_aware_pooling):
    """
    do pooling according to x. a new graph (x, edges) will be generated after pooling.
    This function is a wrapper of some funcs in pytorch_geometric. It assumes the 
    object's coordinates range from -1 to 1.
    
    normal_aware_pooling: if True, only data.x[..., :3] will be used for grid pooling.
    """
    assert type(size) == float
    if normal_aware_pooling == False:
        x = data.x[..., :3]
        # warn the user
    
        #assert data.x[..., :3].max() <= 1.0 and data.x[..., :3].min() >= -1.0
    else:
        x = data.x[..., :6]
        assert data.x[..., :6].max() <= 1.0 and data.x[..., :6].min() >= -1.0, "please normalize the input data to [-1, 1]"
        # we assume x has 6 feature channels, first 3 xyz, then 3 nxnynz
        # grid size here waits for fine-tuning
        n_size = size * global_config.normal_weight   # TODO: a hyper parameter, controling "how important the normal vecs are, compared to xyz coords"
        size = [size, size, size, n_size, n_size, n_size]

    edges = data.edge_index
    cluster = torch_geometric.nn.voxel_grid(pos=x, size=size, batch=None,
                                            start=None, end=None)
                                            #start=[-1, -1, -1.], end=[1, 1, 1.])

    # keep max index smaller than # of unique (e.g., [4,5,4,3] --> [0,1,0,2])
    mapping = cluster.unique()
    mapping += mapping.shape[0]
    cluster += mapping.shape[0]

    for i in range(int(mapping.shape[0])):  
        cluster[cluster == mapping[i]] = i

    # sanity check
    assert cluster.unique().shape[0] <= cluster.shape[0]
    
    return cluster #.contiguous()




def add_self_loop(data: Data):
    # avg_pool will remove self loops, so we need to add self loops back
    device = data.x.device
    n_vert = data.x.shape[0]
    self_loops = torch.tensor([[i for i in range(int(n_vert))]])
    self_loops = self_loops.repeat(2, 1)
    new_edges = torch.zeros([2, data.edge_index.shape[1] + n_vert], dtype=torch.int64)
    new_edges[:, :data.edge_index.shape[1]] = data.edge_index
    new_edges[:, data.edge_index.shape[1]:] = self_loops

    return Data(x=data.x, edge_index=new_edges).to(device)




class GraphTree:
    """
    build a naive graph tree from a graph
    names of arguments and functions are self-explainable

    notes:
    - coordinates of input vertices should be in [-1, 1]
    - this class ONLY handles xyz coordinates, it does NOT handle features

    """
    def __init__(self, depth: int=5,
                 smallest_grid=2/2**5,
                 batch_size: int=1,
                 ):
        """
        For simplicity, the setup is as follows:

        Assuming depth=3, self.graphtree provides access to the following:
        self.graphtree[0] = original graph, type: Data
        self.graphtree[1] = graph merged with the smallest_grid as the grid size, type: Data
        self.graphtree[2] = graph merged with smallest_grid * (2**1) as the grid size, type: Data
        self.graphtree[3] = graph merged with smallest_grid * (2**2) as the grid size, type: Data

        __init__ only defines a hyperparameter for graphtree; the actual construction function should be either build_single_graphtree or merge_graphtree.
        """

        assert smallest_grid * (2**(depth-1)) <= 2, "In such hyperparameter setting, the largest grid size should be bigger than 2 (the whole space)"
        assert depth >= 0

        self.depth = depth
        self.batch_size = batch_size
        self.smallest_grid = smallest_grid
        self.normal_aware_pooling = global_config.normal_aware_pooling

        self.mesh_pooling = PoolingGraph() # deprecated
        self.mesh_unpooling = UnpoolingGraph()
        
    
        self.vertices_sizes = {}
        self.edges_sizes = {}
        #
        self.treedict = {}
        self.cluster = {}
        
        self.prolongation_matrices = {}  # deprecated and not used
        self.stiffness_matrices = {}  # L of Laplacian
        self.mass_matrices = {}  # M of Laplacian
        
        self.layers_eigen_vecs = {}  # eigen vectors.
        self.layers_eigen_vals = {}  # eigen values.
        self.current_eigen_vecs = {} # deprecated and not used
        
        self.base_eigen_val_pred = None     # deprecated and not used
        self.knn_eigen_vec_pred = None      # deprecated and not used
        self.original_mesh = None           # a list of original meshes. used for reference during visualization
        
        #self.pooling = PoolingGraph()
        
        self.original_mesh = []
        self.coarse_meshes = []
        
        self.edge_weight = None             
        self.vertex_mass = None             
        self.real_underlying_graph = None   # a Data object, representing the real underlying graph of the original mesh
        
        # debug 
        self.mesh_name = []
      

    def build_single_graphtree(self, 
                               original_graph: Data, 
                               original_mesh: trimesh.Trimesh=None,
                               laplacian: list[torch.sparse_coo_tensor]=None, 
                               mass_matrix: list[torch.sparse_coo_tensor]=None,
                               real_eigen_vecs=None, 
                               real_eigen_vals=None,
                               real_underlying_graph: Data=None,
                               mesh_name: str=""
                               ):
        """
        build a graph-tree of **one** graph
        """

        self.treedict = {}
        self.cluster = {}
        self.vertices_sizes = {}
        self.edges_sizes = {}
        self.original_mesh.append(original_mesh)
        self.mesh_name.append(mesh_name)
        
        
        # construct the hierarchy of graphs, and the mapping between adjacent layers
        for i in range(self.depth+1):
            # i=0 for the original graph
            if i == 0:
                #original_graph = add_self_loop(original_graph)
                # if original graph do not have edge types, assign it 
                if original_graph.edge_attr == None:
                    edges = original_graph.x[original_graph.edge_index[0]] \
                            - original_graph.x[original_graph.edge_index[1]]
                    #edges_attr = decide_edge_type_distance(edges, return_edge_length=False)
                    #original_graph.edge_attr = edges_attr
                self.treedict[0] = original_graph
                self.cluster[0] = None
                self.edges_sizes[0] = original_graph.edge_index.shape[1]
                self.vertices_sizes[0] = original_graph.x.shape[0]
                # initial stiffness matrix and mass matrix
                self.stiffness_matrices[0] = []
                self.mass_matrices[0] = []
                if laplacian is not None:
                    self.stiffness_matrices[0].append(torch.sparse_coo_tensor(laplacian[0].cpu(), laplacian[1], laplacian[2]).to(torch.float32))
                if mass_matrix is not None:
                    self.mass_matrices[0].append(torch.sparse_coo_tensor(mass_matrix[0].cpu(), mass_matrix[1], mass_matrix[2]).to(torch.float32))
                
                self.layers_eigen_vals[0] = []
                self.layers_eigen_vals[0].append(real_eigen_vals if real_eigen_vals is not None else None)  # real_eigen_vals: [K]
                self.layers_eigen_vecs[0] = []
                self.layers_eigen_vecs[0].append(real_eigen_vecs if real_eigen_vecs is not None else None) # real_eigen_vecs: [N, K]
                
                continue

            # i>=1, graphs after pooling
            clst = pooling(self.treedict[i-1], self.smallest_grid * (2**(i-1)), normal_aware_pooling=self.normal_aware_pooling)
            new_graph = avg_pool(cluster=clst, data=self.treedict[i-1], transform=None)
            #new_graph = add_self_loop(new_graph)
            # assign edge type
            edges = new_graph.x[new_graph.edge_index[0]] \
                    - new_graph.x[new_graph.edge_index[1]]
            #edges_attr = decide_edge_type_distance(edges, return_edge_length=False)
            #new_graph.edge_attr = edges_attr

            self.treedict[i] = new_graph
            self.cluster[i] = clst
            self.edges_sizes[i] = new_graph.edge_index.shape[1]
            self.vertices_sizes[i] = new_graph.x.shape[0]
        
        # real_underlying_graph
        if real_underlying_graph is not None:
            self.real_underlying_graph = real_underlying_graph
        
        # store the edge weight and vertex mass
        if laplacian is not None:
            edge_a_idx = real_underlying_graph.edge_index[0]
            edge_b_idx = real_underlying_graph.edge_index[1]
    
            
            # extract the vertex mass and edge weight
            self.vertex_mass = np.zeros(real_underlying_graph.x.shape[0])
            for i in range(real_underlying_graph.x.shape[0]):
                self.vertex_mass[i] = self.mass_matrices[0][0][i, i].item()
            
            self.edge_weight = np.zeros(len(edge_a_idx))
            from tqdm import tqdm
            for i in tqdm(range(len(edge_a_idx))):
                self.edge_weight[i] = self.stiffness_matrices[0][0][edge_a_idx[i], edge_b_idx[i]].item()
            
            self.vertex_mass = torch.tensor(self.vertex_mass, dtype=torch.float32)
            self.edge_weight = torch.tensor(self.edge_weight, dtype=torch.float32)


    # @staticmethod
    #@profile
    def merge_graphtree(self, original_graphs: list[GraphTree], debug_report=False):
        """
        Note that this function can only merge multiple single graph trees. A graphtree that has already been merged cannot be merged again.
        """
        assert len(self.cluster) == 0 and len(self.treedict) == 0, "A graphtree that has already been merged cannot be merged again."
        assert original_graphs.__len__() == self.batch_size, "The number of original graphs should be equal to the batch size."
        

        # re-indexing
        for d in range(self.depth+1):
            # merge at each layer
            num_vertices = [0]
            for i, each in enumerate(original_graphs):
                num_vertices.append(each.vertices_sizes[d])
            cum_sum = torch.cumsum(torch.tensor(num_vertices), dim=0)
            for i in range(original_graphs.__len__()):
                original_graphs[i].treedict[d].edge_index += cum_sum[i]
                # the 0 layer does not have cluster, so we do not need to process it
                if d != 0:
                    original_graphs[i].cluster[d] += cum_sum[i]

        # merge 
        for d in range(self.depth+1):
            graphtrees_x, graphtrees_e, graphtrees_e_type, clusters = [], [], [], []
            batching = []
            for i in range(original_graphs.__len__()):
                graphtrees_x.append(original_graphs[i].treedict[d].x)
                graphtrees_e.append(original_graphs[i].treedict[d].edge_index)
                graphtrees_e_type.append(original_graphs[i].treedict[d].edge_attr)
                clusters.append(original_graphs[i].cluster[d])
                # batching information
                batching.append(torch.ones([original_graphs[i].treedict[d].x.shape[0]], dtype=torch.int32) * i)
                
                
            # construct new graph
            temp_x = torch.cat(graphtrees_x, dim=0).float()
            temp_edge_index = torch.cat(graphtrees_e, dim=1)
            temp_batching = torch.cat(batching, dim=0)
            temp_edges_size_of_each_subgraph = [original_graphs[i].treedict[d].edge_index.shape[1] for i in range(original_graphs.__len__())]
            temp_data = Data(
                             x=temp_x,          # convert to float32
                             edge_index=temp_edge_index,
                             batch=temp_batching,
                             edges_size_of_each_subgraph=temp_edges_size_of_each_subgraph,
                             subgraphs={"x": graphtrees_x, "edge_index": graphtrees_e}
                             )
            
            if d != 0:
                temp_clst = torch.cat(clusters, dim=0)
            else:
                temp_clst = None
            self.treedict[d] = temp_data
            self.cluster[d] = temp_clst
            self.edges_sizes = temp_data.edge_index.shape[1]
            self.vertices_sizes = len(temp_data.x)
            
        
        
        # concat original mesh
        self.original_mesh = []
        for i in range(len(original_graphs)):
            self.original_mesh.append(original_graphs[i].original_mesh[0])
            
        # concat edge_weight
        temp_edge_weight = []
        for i in range(len(original_graphs)):
            temp_edge_weight.append(original_graphs[i].edge_weight)            
        self.edge_weight = torch.cat(temp_edge_weight, dim=0) if temp_edge_weight[0] is not None else None
        
        # concat vertex_mass
        temp_vertex_mass = []
        for i in range(len(original_graphs)):
            temp_vertex_mass.append(original_graphs[i].vertex_mass)
        self.vertex_mass = torch.cat(temp_vertex_mass, dim=0) if temp_vertex_mass[0] is not None else None
        
        # concat real underlying graph
        if original_graphs[0].real_underlying_graph is not None:
            # re-indexing
            num_vertices = [0]
            for i, each in enumerate(original_graphs):
                num_vertices.append(each.real_underlying_graph.x.shape[0])
            cum_sum = torch.cumsum(torch.tensor(num_vertices), dim=0)
            edge_indices = []
            for i in range(original_graphs.__len__()):
                edge_indices.append(original_graphs[i].real_underlying_graph.edge_index + cum_sum[i])
                #original_graphs[i].real_underlying_graph.edge_index += cum_sum[i]
            temp_underlying_graph_x = [original_graphs[i].real_underlying_graph.x for i in range(len(original_graphs))]
            temp_underlying_graph_edge_index = torch.cat(edge_indices, dim=1)
            temp_data = Data(x=torch.cat(temp_underlying_graph_x, dim=0).float(),          # convert to float32
                            edge_index=temp_underlying_graph_edge_index,
                            )
            self.real_underlying_graph = temp_data
            assert self.real_underlying_graph.edge_index.max() == self.real_underlying_graph.x.shape[0] - 1
        else:
            #
            self.real_underlying_graph = None
        
        # concat name
        self.mesh_name = []
        for i in range(len(original_graphs)):
            self.mesh_name.append(original_graphs[i].mesh_name[0])
            
        # concat layers_eigen_vecs/layers_eigen_vals
        if global_config.store_eigens_of_lap_mat:
            self.layers_eigen_vecs = {0: []}
            self.layers_eigen_vals = {0: []}
            for i in range(len(original_graphs)):
                if 0 in original_graphs[i].layers_eigen_vecs.keys():
                    eigen_vec = original_graphs[i].layers_eigen_vecs[0][0]
                    self.layers_eigen_vecs[0].append(eigen_vec)
                    eigen_val = original_graphs[i].layers_eigen_vals[0][0]
                    self.layers_eigen_vals[0].append(eigen_val)

        # sanity check
        if debug_report == True:
            # a simple unit test
            for d in range(self.depth+1):
                num_edges_before = 0
                for i in range(original_graphs.__len__()):
                    num_edges_before += original_graphs[i].treedict[d].edge_index.shape[1]
                num_edges_after = self.treedict[d].edge_index.shape[1]
                print(f"Before merge, at d={d} there's {num_edges_before} edges; {num_edges_after} afterwards")

        assert self.cluster[1].min() == 0
        for i in range(5):
            assert self.treedict[i].edge_index.max() < self.treedict[i].x.shape[0]

    #####################################################
    # for debugging
    #####################################################
    
    def flip_subspace_eigens(self):
        # this function is used to flip the sign of the eigens in the subspace
        for i in self.depth:
            if self.subspace_eigens[i] != None:
                self.subspace_eigens[i] = - self.subspace_eigens[i]

    #####################################################
    # Util
    #####################################################
    
    
    def index_correction(self):
        # 2023.9.13
        # when batch != 1, pyg seems not to handle the index as we expected, so a re-indexing is needed
        error_count = 0
        for i in self.cluster.keys():
            if self.cluster[i] is not None and self.cluster[i].min() != 0:
                self.cluster[i] -= self.cluster[i].min()
                error_count += 1
        for i in self.treedict.keys():
            if self.treedict[i] is not None and self.treedict[i].edge_index.min() != 0:
                self.treedict[i].edge_index -= self.treedict[i].edge_index.min()
                error_count += 1
      #  print(f"{error_count} errors have been corrected!")
        return
        
    
    
    
    def dumps(self):
        raise NotImplementedError
    
    def to_numpy(self):
        raise NotImplementedError
    
    def to_tensor(self):
        raise NotImplementedError
    
    def cuda(self):
        return self.to("cuda")
        
    def cpu(self):
        return self.to("cpu")
    
    #@profile
    def to(self, device:str):
        # TODO: check if we forget to move some tensors to device
        
        for each in self.treedict.keys():
            self.treedict[each] = self.treedict[each].to(device)
        for each in self.cluster.keys():
            if self.cluster[each] is None:
                continue
            self.cluster[each] = self.cluster[each].to(device)
                
        for each in self.layers_eigen_vals.keys():
            for i in range(len(self.layers_eigen_vals[each])):
                if self.layers_eigen_vals[each][i] is None:
                    continue
                self.layers_eigen_vals[each][i] = self.layers_eigen_vals[each][i].to(device)
                
        for each in self.layers_eigen_vecs.keys():
            for i in range(len(self.layers_eigen_vecs[each])):
                if self.layers_eigen_vecs[each][i] is None:
                    continue
                self.layers_eigen_vecs[each][i] = self.layers_eigen_vecs[each][i].to(device)

        if self.base_eigen_val_pred != None:
            self.base_eigen_val_pred = self.base_eigen_val_pred.to(device)
        if self.knn_eigen_vec_pred != None:
            self.knn_eigen_vec_pred = self.knn_eigen_vec_pred.to(device)
            
        # lap edge weight
        if self.edge_weight != None:
            self.edge_weight = self.edge_weight.to(device)
        # vertex mass
        if self.vertex_mass != None:
            self.vertex_mass = self.vertex_mass.to(device)
        # real underlying graph
        if self.real_underlying_graph != None:
            self.real_underlying_graph = self.real_underlying_graph.to(device)

        return self



 





def construct_graph_tree_from_mesh(vertices, faces, K=10):
    """
    construct graph tree from triangle mesh
    """
    
    raise NotImplementedError
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    vertices = mesh.vertices
    faces = mesh.faces
    # 
    vertices = (vertices - vertices.min()) / (vertices.max() - vertices.min()) * 2 - 1
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # compute the laplacian eigenvalues and eigenvectors, so that:
    # (laplacian @ eigenvector) - (eigenvalue * mass_matrix @ eigenvector) = 0
    
    # compute the laplacian and mass matrix. note that they are sparse, so we save them in COO format
    laplacian, mass_matrix = get_GT_L_and_M(vertices, faces)
    sparse_laplacian = coo_matrix(laplacian)
    sparse_mass_matrix = coo_matrix(mass_matrix)
    #sp.save_npz(os.path.join(output_folder, os.path.basename(mesh_path) + "_sparse_laplacian.npz"), sparse_laplacian)
    #sp.save_npz(os.path.join(output_folder, os.path.basename(mesh_path) + "_sparse_mass_matrix.npz"), sparse_mass_matrix)
    
    # compute the eigenvalues and eigenvectors
    try:
        pass
       # eigenvalues, eigenvectors = get_GT_eigens_of_mesh(vertices, faces, K=K)   
        #   np.save(os.path.join(output_folder, os.path.basename(mesh_path) + "_eigenvalues.npy"), eigenvalues)
        #  np.save(os.path.join(output_folder, os.path.basename(mesh_path) + "_eigenvectors.npy"), eigenvectors)
    except:
        print("Error encountered when computing eigenvalues and eigenvectors.")
        return None
    
    graph_tree = GraphTree()
    graph_tree.build_single_graphtree(
        original_graph=convert_mesh_to_graph_data(vertices, faces),
        original_mesh=mesh,
        laplacian=[torch.LongTensor([sparse_laplacian.row, sparse_laplacian.col]), sparse_laplacian.data, torch.Size(sparse_laplacian.shape)],
        mass_matrix=[torch.LongTensor([sparse_mass_matrix.row, sparse_mass_matrix.col]), sparse_mass_matrix.data, torch.Size(sparse_mass_matrix.shape)],
        real_eigen_vals=None,#torch.tensor(eigenvalues, dtype=torch.float32),
        real_eigen_vecs=None,#torch.tensor(eigenvectors, dtype=torch.float32),
        real_underlying_graph=convert_mesh_to_graph_data(vertices, faces),
    )
    graph_tree = graph_tree.cpu()
    return graph_tree


def construct_graph_tree_from_point_cloud(vertices, ref_mesh:trimesh.Trimesh=None, mesh_name:str=""):
    """
    generate a graph tree from a point cloud

    vertices: shape: [N, 3]
    ref_mesh: a reference mesh which could provide ground truth Laplacian.
    mesh_name: for debugging, record the name of the mesh so we can identify it later.
    """
    
    laplacian = None
    mass_matrix = None
    normal = None
    real_eigen_vals = None
    real_eigen_vecs = None
    
    mesh_vertices, mesh_faces = None, None
    sparse_laplacian_final = None
    sparse_mass_matrix_final = None
    real_eigen_vals = None
    real_eigen_vecs = None
    real_underlying_graph = None
    
 
    if global_config.use_ref_mesh == True:
        if ref_mesh is None:
            raise ValueError("ref_mesh is None.")
        mesh_vertices, mesh_faces = ref_mesh.vertices, ref_mesh.faces
        mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)
        normal = mesh.vertex_normals
        
    
        # laplacian, mass matrix from get_GT_L_and_M(): [N, N] dense matrix
        laplacian, mass_matrix = get_GT_L_and_M(mesh_vertices, mesh_faces)
        # get the "relative mass"
        mass_matrix = relative_mass_matrix(M=mass_matrix, smallest_limit=0.01)
        # convert them to sparse matrix to save memory
        sparse_laplacian = coo_matrix(laplacian)
        sparse_mass_matrix = coo_matrix(mass_matrix)
        temp_lap = np.array([sparse_laplacian.row, sparse_laplacian.col])
        sparse_laplacian_final = [torch.LongTensor(temp_lap), sparse_laplacian.data, torch.Size(sparse_laplacian.shape)]
        temp_mass = np.array([sparse_mass_matrix.row, sparse_mass_matrix.col])
        sparse_mass_matrix_final = [torch.LongTensor(temp_mass), sparse_mass_matrix.data, torch.Size(sparse_mass_matrix.shape)]
    
        if global_config.store_eigens_of_lap_mat:
            # get the eigenvalues and eigenvectors of the laplacian matrix. we may need or not need the mass matrix, depending on the config
            # Two options to get eigens:
            # 1. use get_GT_eigens_of_mesh(verts, faces). this is the most accurate way, but it is NOT good if relative_mass_matrix is used
            # 2. use get_GT_eigens_of_mesh(L=laplacian, M=mass_matrix). this is the most accurate way, but it is NOT good if relative_mass_matrix is used
            get_eigen = 2
            if get_eigen == 1:
                # 1.
                real_eigen_vals, real_eigen_vecs = get_GT_eigens_of_mesh(mesh_vertices, 
                                                                         mesh_faces, 
                                                                         K=global_config.eigens_of_lap_mat_num, 
                                                                         consider_mass_matrix=global_config.use_vertex_mass,
                                                                         method=1
                                                                         )
            elif get_eigen == 2:
                # 2. 
                real_eigen_vals, real_eigen_vecs = get_GT_eigens_of_mesh(L=laplacian,
                                                                         M=mass_matrix,
                                                                         K=global_config.eigens_of_lap_mat_num, 
                                                                         consider_mass_matrix=global_config.use_vertex_mass,
                                                                         method=2
                                                                         )
            else:
                raise NotImplementedError
        
    
        real_underlying_graph = convert_mesh_to_graph_data(mesh_vertices, mesh_faces)
    
    # 
    else:
        pass
    
    graph_tree = GraphTree()
    
    

    original_graph = convert_point_cloud_to_graph(vertices, normal=normal, K=global_config.point_cloud_knn_k)
        
    
    graph_tree.build_single_graphtree(
        original_graph=original_graph,
        original_mesh=ref_mesh,
        laplacian=sparse_laplacian_final,
        mass_matrix=sparse_mass_matrix_final,
        real_eigen_vals=real_eigen_vals,
        real_eigen_vecs=real_eigen_vecs,
        real_underlying_graph=real_underlying_graph,
        mesh_name=mesh_name,
    )
    graph_tree = graph_tree.cpu()
    return graph_tree

