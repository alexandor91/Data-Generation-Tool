'''
This is a vtk library to visualize point cloud
from multiple depth maps aligned with camera settings.

Author: yn
Date: Jan 2020
'''
import sys
sys.path.append('./')
from data_config import shapenet_rendering_path, total_view_nums
import os
from pc_painter import PC_from_DEP
from data_config import camera_setting_path

if __name__ == '__main__':
    depth_sample_dir = '02818832/e91c2df09de0d4b1ed4d676215f46734' # back sofa chair
    #'03001627/7ee5785d8695cf0ee7c7920f6a65a54d'  # thin chair
    #'03001627/ffd9387a533fe59e251990397636975f' # thih chair
    #'02818832/e91c2df09de0d4b1ed4d676215f46734' # bed 
    #'02818832/f7edc3cc11e8bc43869a5f86d182e67f' #bed 

    n_views = 2
    assert n_views <= total_view_nums # there are total 20 views surrounding an object.
    view_ids = range(1, n_views+1)
    metadata_dir = os.path.join(shapenet_rendering_path, depth_sample_dir)
    #print("&&&&&&&&&&&")
    #print(metadata_dir)
    pc_from_dep = PC_from_DEP(metadata_dir, camera_setting_path, view_ids, with_normal=False)
    pc_from_dep.draw_depth(view='all')
    #pc_from_dep._point_clouds
    # pc_from_dep.draw_color(view='all')
    #pc_from_dep.draw3D()