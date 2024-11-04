import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import igl
import hashlib

import robust_laplacian
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import scipy.sparse.linalg as sla












def get_GT_L_and_M(vertices, faces):
    # given a mesh, computes its laplacian and mass matrix
    # where L @ x = w * M @ x
    # return: L: [N, N], M: [N, N], where L is sparse, and M is diagonal
    
    # remember to convert them to numpy array
    
    L, M = robust_laplacian.mesh_laplacian(np.array(vertices), np.array(faces))
  #  M = coo_matrix(M)
  #  L = coo_matrix(L)
    
    
 #   L = - igl.cotmatrix(vertices, faces).toarray()
  #  M = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_VORONOI).toarray()
    
    return L, M
    



def get_GT_eigens_of_mesh(
    vertices=None, 
    faces=None,
    K=3, 
    L=None,
    M=None, 
    method=1,
    consider_mass_matrix=True
    ):
    """
    给定一个网格，计算其前 K 个最小的特征值及其对应的特征向量。提供两种计算方法：
    1. 给定顶点和面片，直接计算。速度快，准确性佳，但是在使用了 relative_mass 时不能使用。
    2. 给定拉普拉斯矩阵和质量矩阵，当作广义特征值问题求解。
    
    vertices: np.ndarray, [N, 3]
    faces: np.ndarray, [M, 3]
    K: int, the first K eigens will be computed (the all-zero eigen will be ignored)
    L: np.ndarray, [N, N], the laplacian matrix of the mesh
    M: np.ndarray, [N, N], the mass matrix of the mesh
    method: 1 or 2，对应上述两种方法。
    consider_mass_matrix: 是否考虑质量矩阵。如果不考虑，那么 M 就是单位阵。
    
    return: eigenvalues: torch.tensor [K], eigenvectors: torch.tensor [N, K]
    """
    
    assert method == 1 or method == 2
    
    # 根据顶点和面，计算特征值和特征向量。此处使用稀疏矩阵进行计算。
    if method == 1:
        L = - igl.cotmatrix(vertices, faces)
        M = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_VORONOI)
        # if encountered numerical difficulties, try the following line:
        # L, M = robust_laplacian.mesh_laplacian(np.array(vertices), np.array(faces))
        
        # for sparse matrix, we recommand use sla.eigsh instead of trying to solve eigen problem
        # using np.linalg.eig, since it may lead to numerical errors
        # A @ x[i] = w[i] * M @ x[i]
        if consider_mass_matrix:
            eigenvalues, eigenvectors = sla.eigsh(L, K+1, M, sigma=1e-6)
        else:
            raise NotImplementedError("please use the mass matrix.")
            eigenvalues, eigenvectors = sla.eigsh(L, K+1, sigma=1e-6)
        # 删除第一个全 0 的特征值和特征向量
        assert eigenvalues[0] <= 1e-5 and eigenvalues[0] >= -1e-5
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]
        # convert to tensor
        eigenvalues = torch.from_numpy(eigenvalues)
        eigenvectors = torch.from_numpy(eigenvectors)
        return eigenvalues, eigenvectors
        
        
    # 根据拉普拉斯矩阵和质量矩阵，直接计算特征值和特征向量。此处使用稠密矩阵进行计算。
    # 此处若使用 sla.eigsh，会出现如下错误：Intel MKL ERROR: Parameter 4 was incorrect on entry to SLASCL.
    elif method == 2:
        
        if consider_mass_matrix:
            eigenvalues, eigenvectors = sla.eigsh(L, K+1, M, sigma=1e-6)
        else:
            raise NotImplementedError("please use the mass matrix.")
            eigenvalues, eigenvectors = sla.eigsh(L, K+1, sigma=1e-6)
        # 删除第一个全 0 的特征值和特征向量
        assert eigenvalues[0] <= 1e-5 and eigenvalues[0] >= -1e-5
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]
        # convert to tensor
        eigenvalues = torch.from_numpy(eigenvalues)
        eigenvectors = torch.from_numpy(eigenvectors)
        return eigenvalues, eigenvectors
        # LPBPCG 速度太慢。20240102，用 scipy.sparse.linalg.eigsh 代替。
        
        
        """
        # LPBPCG algorithm is not applicable when the number of A rows (=L.shape[0]) is smaller 
        # than 3 x the number of requested eigenpairs (=K)
        # 有时，经过 pooling 得到的方阵 L，M 的大小较小，而 K 又较大，导致无法得到 K 个特征值。
        K_original = K
        if L.shape[0] <= K * 3:
            raise ValueError("Legacy code, not applicable now.")
            K = L.shape[0] // 3 - 1
        # convert to tensor
        if isinstance(L, np.ndarray):
            L = torch.from_numpy(L)
        if isinstance(M, np.ndarray):
            M = torch.from_numpy(M)
            
        # mass matrix is diagonal
        if consider_mass_matrix == False:
            M = torch.eye(L.shape[0])

        eigenvalues, eigenvectors = torch.lobpcg(
                                                A=L,
                                                B=M,
                                                k=K+1,
                                                X=None,
                                                largest=False,
                                                niter=600,
                                                tol=1e-7,
                                                method="ortho",
                                                )
        
        # lobpcg 计算的第一个特征值往往没有那么小...所以就让我们相信lobpcg，
        #assert eigenvalues[0] <= 1e-5 and eigenvalues[0] >= -1e-5
        
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]
        
        # padding
        if eigenvectors.shape[1] < K_original:
            print(eigenvectors.shape[1], K_original)
            raise ValueError("Legacy code, not applicable now.")
            eigenvectors = torch.cat([eigenvectors, torch.zeros(eigenvectors.shape[0], K_original - eigenvectors.shape[1])], dim=1)
            eigenvalues = torch.cat([eigenvalues, torch.zeros(K_original - eigenvalues.shape[0])], dim=0)

        
        # eigenvalues: [K]
        # eigenvectors: [num_verts, K]
        # both are sorted by eigenvalues, from small to large
        return eigenvalues, eigenvectors
        """
    



def check_closeness(array:torch.Tensor, index:int, threshold=0.98):
    '''
    这个函数拿到一个 1D array，查看index位置的数字与相邻的左边右边的数字有多么接近。
    如果与任意一者的比率在 threshold 以下，就返回 True，反之返回 False
    '''
    if array.__len__() == 1: 
        return False 
    ratio = 0.0001
    if index >= 0:
        ratio_temp = min(array[index]/array[index-1], array[index-1]/array[index])
        ratio = max(ratio, ratio_temp)
    if index < array.__len__()-1:
        ratio_temp = min(array[index]/array[index+1], array[index+1]/array[index])
        ratio = max(ratio, ratio_temp)
    # make the judgement
    #print(ratio, ratio > threshold)
    if ratio > threshold:
        return True
    else:
        return False




def add_noise_to_vertices_position(vertices: np.ndarray,
                                   strength: float = 0.02,
                                   ):
    """
    对于给定的顶点位置，以高斯噪声的方式随机移动每个顶点的位置，以增加数据的多样性。
    此函数会确保顶点的位置仍然在 [-1, 1] 之间。
    """

    assert vertices.shape[1] == 3 and vertices.shape[0] > 0
    noise = np.random.normal(loc=0.0, scale=strength, size=vertices.shape)
    # 确保顶点的位置仍然在 [-1, 1] 之间
    vertices = vertices + noise
    vertices = np.clip(vertices, -1, 1)
    return vertices







def check_if_exist_multi_edge(graph):
    """
    给定一个图，检查是否存在重边。注意此函数是暴力检查，时间复杂度为 O(E^2)，没有必要的话不要使用。
    """
    edges = graph.edge_index
    for i in range(edges.shape[1]):
        for j in range(i+1, edges.shape[1]):
            if edges[0][i] == edges[0][j] and edges[1][i] == edges[1][j]:
                return True
    return False




def convert_dense_tensor_to_sparse_numpy(torch_tensor: torch.Tensor) -> coo_matrix:
    """
    将一个稠密的 tensor 转换为一个稀疏的 scipy.sparse.coo_matrix
    """
    np_array = torch_tensor.detach().cpu().numpy()
    sparse_matrix = coo_matrix(np_array)
    return sparse_matrix



def convert_sparse_numpy_to_dense_tensor(sparse_matrix: coo_matrix) -> torch.Tensor:
    """
    将一个稀疏的 scipy.sparse.coo_matrix 转换为一个稠密的 torch.Tensor
    """
    np_array = sparse_matrix.toarray()
    torch_tensor = torch.from_numpy(np_array)
    return torch_tensor



def relative_mass_matrix(M: np.ndarray, smallest_limit: float = 0.1):
    """
    给定一个对角阵 M（可以是稠密阵也可以是稀疏阵），计算其相对质量矩阵，即 M / M.mean()
    M: [N, N]
    smallest_limit: 若为浮点数，则返回时将所有的质量限制在 [smallest_limit, inf] 之间。若为 None，则不进行限制。
    返回：一个COO格式的稀疏矩阵
    """

    masses = M.diagonal()
    mean_mass = masses.mean()
    relative_masses = masses / mean_mass
    
    if smallest_limit:
        # 即，我们假设最小的顶点，其质量也有 smallest_limit 个平均质量。
        # 如果不这么做，那么最小的顶点的质量可能会很小，导致其逆矩阵的值很大，造成数值不稳定。
        relative_masses = np.maximum(relative_masses, smallest_limit)
        #relative_masses = torch.clamp(relative_masses, min=smallest_limit)
        
    # 创建一个对角阵
    M_realtive = sp.diags(relative_masses, 0, format='coo')
    #M_realtive = np.diag(relative_masses)       # TODO: 重复创建大矩阵可能不是很好
    return M_realtive
    

