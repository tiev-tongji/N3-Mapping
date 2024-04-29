import os
import os.path as osp
import sys
sys.path.append(osp.abspath('./'))

import copy
import numpy as np
from numpy.linalg import inv, norm
import torch
from torch.utils.data import Dataset
import open3d as o3d
from natsort import natsorted 

from utils.config import SHINEConfig
from utils.pose import *
from utils.semantic_kitti_utils import *
from utils.tools import voxel_down_sample_torch


class KITTIDataset(Dataset):
    def __init__(self, config: SHINEConfig) -> None:

        super().__init__()

        self.config = config
        self.dtype = config.dtype
        torch.set_default_dtype(self.dtype)
        self.device = config.device
        self.pool_device = config.device

        self.calib = {}
        if config.calib_path != '':
            self.calib = read_calib_file(config.calib_path)
        else:
            self.calib['Tr'] = np.eye(4)
        if config.pose_path.endswith('txt'):
            self.poses_w = read_poses_file(config.pose_path, self.calib)
        elif config.pose_path.endswith('csv'):
            self.poses_w = csv_odom_to_transforms(config.pose_path)
        else:
            sys.exit(
            "Wrong pose file format. Please use either *.txt (KITTI format) or *.csv (xyz+quat format)"
            )

        # pose in the reference frame (might be the first frame used)
        self.poses_ref = self.poses_w  # initialize size

        # point cloud files
        self.pc_filenames = natsorted(os.listdir(config.pc_path)) # sort files as 1, 2,… 9, 10 not 1, 10, 100 with natsort
        self.total_pc_count = len(self.pc_filenames)

        # local map pc
        self.cur_frame_pc = o3d.geometry.PointCloud()
        # merged downsampled point cloud
        self.map_down_pc = o3d.geometry.PointCloud()
        # map bounding box in the world coordinate system
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()

        # get the pose in the reference frame
        self.used_pc_count = 0
        begin_flag = False
        self.begin_pose_inv = np.eye(4)
        for frame_id in range(self.total_pc_count):
            if (
                frame_id < config.begin_frame
                or frame_id > config.end_frame
                or frame_id % config.every_frame != 0
            ):
                continue
            if not begin_flag:  # the first frame used
                begin_flag = True
                if config.first_frame_ref:
                    self.begin_pose_inv = inv(self.poses_w[frame_id])  # T_rw
                else:
                    # just a random number to avoid octree boudnary marching cubes problems on synthetic dataset such as MaiCity(TO FIX)
                    self.begin_pose_inv[2,3] += config.global_shift_default 
            # use the first frame as the reference (identity)
            self.poses_ref[frame_id] = np.matmul(
                self.begin_pose_inv, self.poses_w[frame_id]
            )
            self.used_pc_count += 1
        # or we directly use the world frame as reference

    def process_frame(self, frame_id):

        pc_radius = self.config.pc_radius
        min_z = self.config.min_z
        max_z = self.config.max_z
        normal_radius_m = self.config.normal_radius_m
        normal_max_nn = self.config.normal_max_nn
        rand_down_r = self.config.rand_down_r
        vox_down_m = self.config.vox_down_m
        sor_nn = self.config.sor_nn
        sor_std = self.config.sor_std

        self.cur_pose_ref = self.poses_ref[frame_id]

        # step 0. load point cloud (support *pcd, *ply and kitti *bin format)
        frame_filename = os.path.join(self.config.pc_path, self.pc_filenames[frame_id])
        
        if not self.config.semantic_on:
            frame_pc = self.read_point_cloud(frame_filename)
            # label_filename = os.path.join(self.config.label_path, self.pc_filenames[frame_id].replace('bin','label'))
            # frame_pc = self.read_semantic_point_label(frame_filename, label_filename)
        else:
            label_filename = os.path.join(self.config.label_path, self.pc_filenames[frame_id].replace('bin','label'))
            frame_pc = self.read_semantic_point_label(frame_filename, label_filename)

        #step 1. block filter: crop the point clouds into a cube
        bbx_min = o3d.core.Tensor([-pc_radius, -pc_radius, min_z], dtype = o3d.core.float32)
        bbx_max = o3d.core.Tensor([pc_radius, pc_radius, max_z], dtype = o3d.core.float32)
        bbx = o3d.t.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)
        frame_pc = frame_pc.crop(bbx)

        # surface normal estimation
        if self.config.estimate_normal:
            frame_pc.estimate_normals(max_nn = normal_max_nn)
            #frame_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(normal_max_nn))
            #frame_pc.estimate_normals(radius=normal_radius_m)
            frame_pc.orient_normals_towards_camera_location() # orient normals towards the default origin(0,0,0).


        #step2. point cloud downsampling
        if self.config.rand_downsample:
            # random downsampling
            frame_pc = frame_pc.random_down_sample(sampling_ratio=rand_down_r)
        else:
            # voxel downsampling
            frame_pc = frame_pc.voxel_down_sample(voxel_size=vox_down_m)

        # apply filter (optional)
        if self.config.filter_noise:
            frame_pc = frame_pc.remove_statistical_outlier(
                sor_nn, sor_std, print_progress=False
            )[0]
        
        frame_origin = self.cur_pose_ref[:3, 3] * self.config.scale  # translation part
        frame_origin_torch = torch.tensor(frame_origin, dtype=self.dtype, device=self.pool_device)

        # step 3. transform to reference frame 
        frame_pc = frame_pc.transform(self.cur_pose_ref)

        # step 3.5 make a backup of global point cloud map.
        frame_pc_clone = copy.deepcopy(frame_pc.to_legacy())
        #frame_pc_clone = frame_pc_clone.voxel_down_sample(voxel_size=self.config.map_vox_down_m) # for smaller memory cost
        self.map_down_pc += frame_pc_clone # for marching cube filtering.
        self.cur_frame_pc = frame_pc_clone # for visualization
        self.map_bbx = self.map_down_pc.get_axis_aligned_bounding_box()
        if frame_id % 400 == 0:
            self.map_down_pc = self.map_down_pc.voxel_down_sample(0.5*self.config.mc_res_m) # to avoid out of memory for large map

        # step 4. and scale to [-1,1] coordinate system (important!)
        frame_pc_s = frame_pc.scale(self.config.scale, center=o3d.core.Tensor([0,0,0], dtype = o3d.core.float32))

        # step 5 turn into torch format.
        frame_pc_s_torch = torch.tensor(frame_pc_s.point.positions.numpy(), dtype=self.dtype, device=self.pool_device)
        frame_normal_torch = None
        if self.config.estimate_normal:
            frame_normal_torch = torch.tensor(frame_pc_s.point.normals.numpy(), dtype=self.dtype, device=self.pool_device)
        frame_label_torch = None
        if self.config.semantic_on:
            frame_label_torch = torch.tensor(frame_pc_s.point.labels.numpy(), dtype=self.dtype, device=self.pool_device)
        
        return frame_id, frame_origin_torch, frame_pc_s_torch, frame_normal_torch, frame_label_torch
    
    def read_point_cloud(self, filename: str):
        # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
        if ".bin" in filename:
            points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3]
        elif ".ply" in filename or ".pcd" in filename:
            pc_load = o3d.io.read_point_cloud(filename)
            points = np.asarray(pc_load.points)
        else:
            sys.exit(
                "The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)"
            )
        preprocessed_points = self.preprocess_kitti(points, self.config.min_z, self.config.min_range)
        #preprocessed_points = points
        pcd_t = o3d.t.geometry.PointCloud()
        pcd_t.point.positions = o3d.core.Tensor(preprocessed_points, o3d.core.float32)
        return pcd_t

    def read_semantic_point_label(self, filename: str, label_filename: str):

        # read point cloud (kitti *.bin format)
        if ".bin" in filename:
            points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3]
        elif ".ply" in filename or ".pcd" in filename:
            pc_load = o3d.io.read_point_cloud(filename)
            points = np.asarray(pc_load.points)
        else:
            sys.exit(
                "The format of the imported point cloud is wrong (support only *bin)"
            )

        # read point cloud labels (*.label format)
        if ".label" in label_filename:
            labels = np.fromfile(label_filename, dtype=np.uint32).reshape((-1))
        else:
            sys.exit(
                "The format of the imported point labels is wrong (support only *label)"
            )

        points, sem_labels = self.preprocess_sem_kitti(
            points, labels, self.config.min_z, self.config.min_range, filter_moving=self.config.filter_moving_object
        )
        pcd_t = o3d.t.geometry.PointCloud()
        pcd_t.point.positions = o3d.core.Tensor(points, o3d.core.float32)
        pcd_t.point.labels = o3d.core.Tensor(sem_labels, o3d.core.int32) #.reshape(-1)
        return pcd_t

    def preprocess_kitti(self, points, z_th=-3.0, min_range=2.5):
        # filter the outliers
        z = points[:, 2]
        points = points[z > z_th]
        points = points[np.linalg.norm(points, axis=1) >= min_range]
        return points

    def preprocess_sem_kitti(self, points, labels, min_range=2.75, filter_outlier = True, filter_moving = True):
        # TODO: speed up
        sem_labels = np.array(labels & 0xFFFF)

        range_filtered_idx = np.linalg.norm(points, axis=1) >= min_range
        points = points[range_filtered_idx]
        sem_labels = sem_labels[range_filtered_idx]

        # filter the outliers according to semantic labels
        if filter_moving:
            filtered_idx = sem_labels < 100
            points = points[filtered_idx]
            sem_labels = sem_labels[filtered_idx]

        if filter_outlier:
            filtered_idx = (sem_labels != 1) # not outlier
            points = points[filtered_idx]
            sem_labels = sem_labels[filtered_idx]
        
        sem_labels_main_class = np.array([sem_kitti_learning_map[sem_label] for sem_label in sem_labels]) # get the reduced label [0-20]

        return points, sem_labels_main_class

    def write_merged_pc(self, out_path):
        #map_down_pc_out = copy.deepcopy(self.map_down_pc)
        map_down_pc_out = self.map_down_pc
        map_down_pc_out.transform(inv(self.begin_pose_inv)) # back to world coordinate (if taking the first frame as reference)
        o3d.io.write_point_cloud(out_path, map_down_pc_out) 
        print("save the merged point cloud map to %s\n" % (out_path))
    
    def __len__(self) -> int:
        return len(self.pc_filenames)
    
    def __getitem__(self, index: int):
        return self.process_frame(index)
        
if __name__ == '__main__':
    config = SHINEConfig()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit(
            "Please provide the path to the config file.\nTry: python shine_incre.py xxx/xxx_config.yaml"
        )
    loader = KITTIDataset(config)
    seq_size = len(loader)
    print("the sequence has {0} frames in total.".format(seq_size))
    for frame_id, origin, points, normals, labels in loader:
        print(frame_id)
        print(origin)
        print(points.shape)
        print(normals.shape)
        if(config.semantic_on):
            print(labels.shape)



        


