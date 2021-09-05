'''
Generate watertight mesh from depth scans.
author: ynie
date: Jan, 2020
'''
import sys
sys.path.append('.')
import os
from external import pyfusion
from data_config import shapenet_rendering_path, watertight_mesh_path, camera_setting_path, total_view_nums, shapenet_normalized_path
from tools.read_and_write import read_exr, read_txt, read_json
import numpy as np
import mcubes
from multiprocessing import Pool
from functools import partial
from tools.utils import dist_to_dep
from settings import cpu_cores
from tools.read_and_write import load_data_path
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

voxel_res = 256        # resolution parameter for our voxel size
truncation_factor = 10     # truncation threshold along the watertight surface


# ### 3D View
def plot3Dgrid(grid, az, el):
    # plot the surface
    plt3d = plt.figure(figsize=(1, 1)).gca()

    # create x,y
    ll, bb = np.meshgrid(range(grid.shape[2]), range(grid.shape[1]))

    for z in range(grid.shape[2]):
        #if not (np.max(grid[1,:,:,z])==np.min(grid[1,:,:,z])): # unberührte Ebenen nicht darstellen
        cp = plt3d.contourf(ll, bb, grid[0,:,:,z], offset = z, alpha=0.3, cmap=cm.Greens)

    cbar = plt.colorbar(cp, shrink=0.7, aspect=20)
    cbar.ax.set_ylabel('$P(m|z,x)$')
    
    plt3d.set_xlabel('X')
    plt3d.set_ylabel('Y')
    plt3d.set_zlabel('Z')
    #plt3d.set_xlim3d(0, grid.shape[1])
    #plt3d.set_ylim3d(0, grid.shape[2])
    #plt3d.set_zlim3d(0, grid.shape[3])
    #plt3d.axis('equal')
    #plt3d.view_init(az, el)
    return plt3d

def process_occgrid(obj_path, view_ids, cam_Ks, cam_RTs):
    '''
    script for prepare occupancy grids for training
    :param obj (str): object path
    :param view_ids (N-d list): which view ids would like to render (from 1 to total_view_nums).
    :param cam_Ks (N x 3 x 3): camera intrinsic parameters.
    :param cam_RTs (N x 3 x 3): camera extrinsic parameters.
    :return:
    '''
    #print("$$$$$$$$$$$$$")
    #print(obj_path)
    cat, model = obj_path.split('/')[3:5]

    '''Decide save path'''
    output_file = os.path.join(watertight_mesh_path, cat, model, 'model.off')

    #if os.path.exists(output_file):
    #    return None

    #if not os.path.exists(os.path.join(watertight_mesh_path, cat, model)):
    #    os.makedirs(os.path.join(watertight_mesh_path, cat, model))

    '''Begin to process'''
    #obj_dir = os.path.join(shapenet_rendering_path, cat, model)
    #dist_map_dir = [os.path.join(obj_dir, 'depth_{0:03d}.exr'.format(view_id)) for view_id in view_ids]

    #dist_maps = read_exr(dist_map_dir)
    #depth_maps = np.float32(dist_to_dep(dist_maps, cam_Ks, erosion_size = 2))

    #cam_Rs = np.float32(cam_RTs[:, :, :-1])
    #cam_Ts = np.float32(cam_RTs[:, :, -1])

    #views = pyfusion.PyViews(depth_maps, cam_Ks, cam_Rs, cam_Ts)

    #voxel_size = 1. / voxel_res
    #truncation = truncation_factor * voxel_size
    
    #occ = pyfusion.occupancy_gpu(views, voxel_res, voxel_res, voxel_res, voxel_size, truncation, False)

    #print("@@@@@@@@@@@@@@@@@@@")
    #print(np.shape(occ))  
    #print(occ)  

    #tsdf = pyfusion.tsdf_gpu(views, voxel_res, voxel_res, voxel_res, voxel_size, truncation, False)
    #mask_grid = pyfusion.projmask_gpu(views, voxel_res, voxel_res, voxel_res, voxel_size, False)
    #occ[mask_grid == 0.] = truncation

    # rotate to the correct system
    #occ = np.transpose(occ[0], [2, 1, 0])
    #print(np.shape(occ))  
    #print(occ)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("test-pcl/test.ply")
    #o3d.visualization.draw_geometries([pcd_load])

    # convert Open3D.o3d.geometry.PointCloud to numpy array
    print('voxelization begins!')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_load,
                                                                voxel_size=0.025)
    #o3d.visualization.draw_geometries([voxel_grid]) 
    o3d.io.write_voxel_grid("test-pcl/test.vox", voxel_grid)

    # To ensure that the final mesh is indeed watertight
    #occ = np.pad(tsdf, 1, 'constant', constant_values=1e6)


def process_mesh(obj_path, view_ids, cam_Ks, cam_RTs):
    '''
    script for prepare watertigt mesh for training
    :param obj (str): object path
    :param view_ids (N-d list): which view ids would like to render (from 1 to total_view_nums).
    :param cam_Ks (N x 3 x 3): camera intrinsic parameters.
    :param cam_RTs (N x 3 x 3): camera extrinsic parameters.
    :return:
    '''
    #print("$$$$$$$$$$$$$")
    cat, model = obj_path.split('/')[3:5]

    '''Decide save path'''
    output_file = os.path.join(watertight_mesh_path, cat, model, 'model.off')

    #if os.path.exists(output_file):
    #    return None

    if not os.path.exists(os.path.join(watertight_mesh_path, cat, model)):
        os.makedirs(os.path.join(watertight_mesh_path, cat, model))

    '''Begin to process'''
    obj_dir = os.path.join(shapenet_rendering_path, cat, model)
    dist_map_dir = [os.path.join(obj_dir, 'depth_{0:03d}.exr'.format(view_id)) for view_id in view_ids]

    dist_maps = read_exr(dist_map_dir)
    depth_maps = np.float32(dist_to_dep(dist_maps, cam_Ks, erosion_size = 2))

    cam_Rs = np.float32(cam_RTs[:, :, :-1])
    cam_Ts = np.float32(cam_RTs[:, :, -1])

    views = pyfusion.PyViews(depth_maps, cam_Ks, cam_Rs, cam_Ts)

    voxel_size = 1. / voxel_res
    truncation = truncation_factor * voxel_size
    tsdf = pyfusion.tsdf_gpu(views, voxel_res, voxel_res, voxel_res, voxel_size, truncation, False)
    mask_grid = pyfusion.projmask_gpu(views, voxel_res, voxel_res, voxel_res, voxel_size, False)
    print('mask')
    print(np.shape(mask_grid))  
    if mask_grid.any() == 0. :
        print('!!!!!!!!!!!equal to  zero!!!!!!!!!!!!!')
    tsdf[mask_grid == 0.] = truncation

    # rotate to the correct system
    tsdf = np.transpose(tsdf[0], [2, 1, 0])

    # To ensure that the final mesh is indeed watertight
    tsdf = np.pad(tsdf, 1, 'constant', constant_values=1e6)
    # print('tsdf')
    # print(tsdf)    
    vertices, triangles = mcubes.marching_cubes(-tsdf, 0)
    # Remove padding offset
    vertices -= 1
    # Normalize to [-0.5, 0.5]^3 cube
    vertices /= voxel_res
    vertices -= 0.5

    mcubes.export_off(vertices, triangles, output_file)

if __name__ == '__main__':
    '''generate watertight meshes by patch'''
    all_objects = load_data_path(shapenet_normalized_path)

    '''camera views'''
    view_ids = range(1, total_view_nums + 1)

    '''load camera parameters'''
    cam_K = np.loadtxt(os.path.join(camera_setting_path, 'cam_K/cam_K.txt'))
    cam_Ks = np.stack([cam_K] * total_view_nums, axis=0).astype(np.float32)
    cam_RT_dir = [os.path.join(camera_setting_path, 'cam_RT', 'cam_RT_{0:03d}.txt'.format(view_id)) for view_id in view_ids]
    cam_RTs = read_txt(cam_RT_dir)

    p = Pool(processes=cpu_cores)
    p.map(partial(process_mesh, view_ids=view_ids, cam_Ks=cam_Ks, cam_RTs=cam_RTs), all_objects)
    p.close()
    p.join()