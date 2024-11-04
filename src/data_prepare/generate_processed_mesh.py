import trimesh
import numpy as np
import os
import pymeshlab
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import joblib
import time

from multiprocessing import Pool


"""
Read the meshes in a directory, process them, and save them to another directory.
"""


PROJ_ROOT_DIR = os.path.abspath(os.path.join(os.curdir, "../.."))  # root path of the project



def check_manifold_meshes(mesh: trimesh.Trimesh):
    """
    Check if a mesh is manifold
    """ 
    if not mesh.is_watertight:
        return False
    if not mesh.is_winding_consistent:
        return False
    return True



def process_single_mesh(mesh: trimesh.Trimesh,
                        ):
    """
    1. Scale to -0.9~0.9
    2. Remesh
    """
    
    # move to center
    mesh.apply_translation(-mesh.centroid)
    
    # scale
    mesh.apply_scale(0.9 / np.max(np.abs(mesh.vertices)))
    
    # store a temporary file, because pymeshlab can only load from file. randint may cause collision, so add time.time()
    os.makedirs(os.path.join(PROJ_ROOT_DIR, "data/temp"), exist_ok=True)
    temp_dir = os.path.join(PROJ_ROOT_DIR, "data/temp", np.random.randint(0, 100000000).__str__() + str(time.time()) + ".obj")
    if os.path.exists(temp_dir):        # in case of collision
        os.system("rm {}".format(temp_dir))
    mesh.export( temp_dir )
    
    # load in meshlab format
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(temp_dir)
    
    
    ##############################################################
    # remesh

    REMESHING_FLAG = False
    if REMESHING_FLAG:
        num_faces = ms.current_mesh().face_number()
        # apply QEM
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=num_faces//3)
        # apply isotropic remeshing. Please note that in older version of meshlab, you need "pymeshlab.Percentage" rather than "pymeshlab.PercentageValue". Refer to https://github.com/3DTopia/LGM/issues/2
        ms.apply_filter('meshing_isotropic_explicit_remeshing', iterations=6, targetlen=pymeshlab.PercentageValue(1.75)) # lead to 1000~3000 vertices
        # apply QEM again
        num_faces = ms.current_mesh().face_number()
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=int(num_faces*0.95))
    ##############################################################
    
    ##############################################################
    # check manifoldness
    # not really necessary, but it may help to improve the numerical stability during training
    CHECK_MANIFOLD_FLAG = False   
    if CHECK_MANIFOLD_FLAG:
        temp_trimesh = trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(), faces=ms.current_mesh().face_matrix(), process=False)
        is_manifold = check_manifold_meshes(temp_trimesh)
        #breakpoint()
        if not is_manifold:
            #print(f"Non-manifold mesh detected. ({temp_dir})")
            # delete temporary file
            os.system("rm {}".format(temp_dir))
            return None
        
        # edge flip
        # not really necessary, but it may help to improve the numerical stability during training
        ms.apply_filter('meshing_edge_flip_by_planar_optimization', planartype='delaunay')
    ##############################################################

    # transform to trimesh
    verts = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix()
    out_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # delete temporary file
    os.system("rm {}".format(temp_dir))
    return out_mesh



def process_single_mesh_wrapper(mesh_dir: str,
                                out_dir: str,
                                ):
    """
    Wrapper of process_single_mesh
    """
    mesh = trimesh.load(mesh_dir, process=False)
    processed_mesh = process_single_mesh(mesh)
    if processed_mesh is None:
        return
    else:
        processed_mesh.export(out_dir)
    
    

def multiprocess_process_mesh(mesh_dirs: list,
                              out_dirs: list,
                              num_process: int,
                              ):
    """
    multiprocess process_single_mesh_wrapper
    """
    assert len(mesh_dirs) == len(out_dirs)
    assert num_process > 0
    
    # batch
    num_process = num_process * 4
    batch_num = len(mesh_dirs) // (num_process)
    print("All Batch: {}".format(batch_num), "with each batch {} meshes".format(num_process))
    
    for i in tqdm(range(0, len(mesh_dirs), num_process)):
        temp_mesh_dirs = mesh_dirs[i:i+num_process]
        temp_out_dirs = out_dirs[i:i+num_process]
        
        # create a pool
        pool = Pool(num_process // 4)
    
        for mesh_dir, out_dir in zip(temp_mesh_dirs, temp_out_dirs):
            pool.apply_async(
                process_single_mesh_wrapper, 
                args=(mesh_dir, out_dir)
                )

        # close the pool
        pool.close()
        
        pool.join()
        
    print("Done.")


def main():
    
    SRC_MESH_PATHS = os.listdir(os.path.join(PROJ_ROOT_DIR, "raw_data"))
    # convert to absolute path
    SRC_MESH_PATHS = [os.path.join("raw_data", mesh_path) for mesh_path in SRC_MESH_PATHS]
    OUT_MESH_PATHS = [each.replace("raw_data", "processed_data") for each in SRC_MESH_PATHS]
    
    
    for src_mesh_path, out_mesh_path in zip(SRC_MESH_PATHS, OUT_MESH_PATHS):
        
        # create the output directory
        if os.path.exists(os.path.join(PROJ_ROOT_DIR, out_mesh_path)):
            os.system("rm -r {}".format(os.path.join(PROJ_ROOT_DIR, out_mesh_path)))
        if not os.path.exists(os.path.join(PROJ_ROOT_DIR, out_mesh_path)):
            os.makedirs(os.path.join(PROJ_ROOT_DIR, out_mesh_path))
            
        mesh_names = os.listdir(os.path.join(PROJ_ROOT_DIR, src_mesh_path))
        mesh_names = [mesh_name for mesh_name in mesh_names if mesh_name.endswith(".obj")]
        mesh_names = [os.path.join(PROJ_ROOT_DIR, src_mesh_path, mesh_name) for mesh_name in mesh_names]  # convert to absolute path
        
        print("=====================================================")
        print("Processing {}...".format(src_mesh_path), f"(total:{len(mesh_names)})")
        
        # single process
        for mesh_name in tqdm(mesh_names):
            out_mesh_name = os.path.join(PROJ_ROOT_DIR, out_mesh_path, os.path.basename(mesh_name))
            process_single_mesh_wrapper(mesh_name, out_mesh_name)
        
        # multiprocess
      #  mesh_dirs = mesh_names
     #   out_dirs = [os.path.join(PROJ_ROOT_DIR, out_mesh_path, os.path.basename(mesh_name)) for mesh_name in mesh_names]
      #  multiprocess_process_mesh(mesh_dirs, out_dirs, num_process=48)
        
        
        
if __name__ == "__main__":
    main()