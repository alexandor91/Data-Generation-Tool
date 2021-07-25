'''
This is a vtk library to visualize point cloud
from multiple depth maps aligned with camera settings.

Author: ynie
Date: Jan 2020
'''
import sys
sys.path.append('./')
from data_config import shapenet_rendering_path, total_view_nums
import os
from pc_painter import PC_from_DEP
from data_config import camera_setting_path

if __name__ == '__main__':
    depth_sample_dir = '03001627/7ee5785d8695cf0ee7c7920f6a65a54d'   #chair now
    n_views = 2
    assert n_views <= total_view_nums # there are total 20 views surrounding an object.
    view_ids = range(1, n_views+1)
    metadata_dir = os.path.join(shapenet_rendering_path, depth_sample_dir)
    #print("&&&&&&&&&&&")
    #print(metadata_dir)
    pc_from_dep = PC_from_DEP(metadata_dir, camera_setting_path, view_ids, with_normal=False)
    pc_from_dep.draw_depth(view='all')
    # pc_from_dep.draw_color(view='all')
    pc_from_dep.draw3D()