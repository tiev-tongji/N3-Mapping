import numpy as np
from numpy.linalg import inv, norm
import kaolin as kal
import kaolin.render.spc as spc_render
import torch

from utils.config import SHINEConfig
from utils.tools import *

class dataSampler():

    def __init__(self, config: SHINEConfig):

        self.config = config
        self.dev = config.device
    
    def sampling_rectified_sdf(self, points_torch,  
               sensor_origin_torch,
               normal_torch,
               sem_label_torch):

        dev = self.config.device

        surface_sample_range_scaled = self.config.surface_sample_range_m * self.config.scale
        surface_sample_n = self.config.surface_sample_n
        freespace_sample_n = self.config.free_sample_n
        all_sample_n = surface_sample_n+freespace_sample_n
        free_min_ratio = self.config.free_sample_begin_ratio
        free_sample_end_dist_scaled = self.config.free_sample_end_dist * self.config.scale #meter

        # get sample points
        shift_points = points_torch - sensor_origin_torch
        point_num = shift_points.shape[0]
        distances = torch.linalg.norm(shift_points, dim=1, keepdim=True) # ray distances (scaled)
        ray_direction = shift_points/distances # normalized ray direction
   
        # Part 1. close-to-surface sampling along with normals.
        # uniform sample in the close-to-surface range (+- range) (-1,1)
        surface_sample_ratio_uniform = (torch.rand(point_num*surface_sample_n, 1, device=dev)-0.5)*2
        
        # gaussian sampling (TODO: gaussian should provide both near surface samples and free space samples)
        if self.config.gaussian_sampling_on:
            surface_sample_ratio_gaussian = torch.randn(point_num*surface_sample_n,1,device=dev)*0.3
            condition = torch.logical_and(surface_sample_ratio_gaussian > -1, surface_sample_ratio_gaussian < 1)
            surface_sample_ratio = torch.where(condition, surface_sample_ratio_gaussian, surface_sample_ratio_uniform)
            #print(surface_sample_ratio)
        else:
            surface_sample_ratio = surface_sample_ratio_uniform
        
        surface_sample_displacement = surface_sample_ratio * surface_sample_range_scaled
        repeated_dist = distances.repeat(surface_sample_n,1)
        surface_sample_dist_ratio = surface_sample_displacement/repeated_dist + 1.0 # 1.0 means on the surface

        surface_repeated_points = shift_points.repeat(surface_sample_n,1)
        surface_sample_points = sensor_origin_torch + surface_repeated_points*surface_sample_dist_ratio

        # only near surface samples are assigned to semantic labels.
        if sem_label_torch is not None:
            surface_sem_label_tensor = sem_label_torch.repeat(1, surface_sample_n).transpose(0,1)
        
        # Part 2. free space uniform sampling
        repeated_dist = distances.repeat(freespace_sample_n,1)
        if sem_label_torch is not None:
            free_sem_label_tensor = torch.zeros_like(repeated_dist)

        free_max_ratio = free_sample_end_dist_scaled / repeated_dist + 1.0
        free_diff_ratio = free_max_ratio - free_min_ratio
        free_sample_dist_ratio = torch.rand(point_num*freespace_sample_n, 1, device=dev)*free_diff_ratio + free_min_ratio
        free_sample_displacement = (free_sample_dist_ratio - 1.0) * repeated_dist
        free_repeated_points = shift_points.repeat(freespace_sample_n,1)
        free_sample_points = free_repeated_points*free_sample_dist_ratio + sensor_origin_torch

        all_sample_points = torch.cat((surface_sample_points,free_sample_points),0)
        all_sample_displacement = torch.cat((surface_sample_displacement, free_sample_displacement),0)
        
        weight_tensor = torch.ones_like(all_sample_displacement)
        if self.config.behind_dropoff_on:
            dropoff_min = 0.2 * self.config.scale
            dropoff_max = 0.8 * self.config.scale
            dropoff_diff = dropoff_max - dropoff_min
            dropoff_weight = (dropoff_max - all_sample_displacement) / dropoff_diff
            dropoff_weight = torch.clamp(dropoff_weight, min = 0.1, max = 1.0)
            #print(dropoff_weight)

        # give a flag indicating the type of the sample [negative: freespace, positive: surface]
        weight_tensor[point_num*surface_sample_n:] *= -1.0
        
        # assign sdf labels to the samples
        # projective distance as the label: behind -, in-front +
        sdf_label_tensor = - all_sample_displacement.squeeze(1)

        # assign the normal label to the samples
        normal_label_tensor = None
        if normal_torch is not None:
            normal_label_tensor = normal_torch.repeat(all_sample_n,1).reshape(-1, 3)
        
        # rectify sdf label by normals
        ray_direction_tensor = ray_direction.repeat(all_sample_n,1)
        correct_ratio = (normal_label_tensor * ray_direction_tensor).sum(dim=1).abs()
        sdf_label_tensor = sdf_label_tensor * correct_ratio

        # assign the semantic label to the samples (including free space as the 0 label)
        sem_label_tensor = None
        if sem_label_torch is not None:
            sem_label_tensor = torch.cat((surface_sem_label_tensor, free_sem_label_tensor),0).int().reshape(-1, 3)
        
        # samples to voxel int coords
        all_sample_voxels = kal.ops.spc.quantize_points(all_sample_points, self.config.tree_level_world)
        all_sample_morton = kal.ops.spc.points_to_morton(all_sample_voxels)

        samples = {}
        samples["count"] = sdf_label_tensor.shape[0]
        #samples["point_morton_count"] = point_morton_count
        samples["pcd_count"] = point_num
        samples["coord"] = all_sample_points.reshape(-1,3)
        samples["voxel_coord"] = all_sample_voxels.reshape(-1, 3)
        samples["morton"] = all_sample_morton.reshape(-1)
        samples["sdf"] = sdf_label_tensor.reshape(-1)
        samples["normal"] = normal_label_tensor
        samples["sem"] = sem_label_tensor
        samples["weight"] = weight_tensor.reshape(-1)
        
        return samples
    
    # free space sampling jump near surface
    def sampling(self, points_torch,  
               sensor_origin_torch,
               normal_torch,
               sem_label_torch,
               normal_guided_sampling = False):

        dev = self.config.device

        surface_sample_range_scaled = self.config.surface_sample_range_m * self.config.scale
        surface_sample_n = self.config.surface_sample_n
        freespace_sample_n = self.config.free_sample_n
        all_sample_n = surface_sample_n+freespace_sample_n
        free_min_ratio = self.config.free_sample_begin_ratio
        free_sample_end_dist_scaled = self.config.free_sample_end_dist * self.config.scale #meter

        # get sample points
        shift_points = points_torch - sensor_origin_torch
        point_num = shift_points.shape[0]
        distances = torch.linalg.norm(shift_points, dim=1, keepdim=True) # ray distances (scaled)
   
        # Part 1. close-to-surface sampling
        # uniform sample in the close-to-surface range (+- range) (-1,1)
        surface_sample_ratio_uniform = (torch.rand(point_num*surface_sample_n, 1, device=dev)-0.5)*2
        
        # gaussian sampling (gaussian should provide both near surface samples and free space samples)
        if self.config.gaussian_sampling_on:
            surface_sample_ratio_gaussian = torch.randn(point_num*surface_sample_n,1,device=dev)*0.3
            condition = torch.logical_and(surface_sample_ratio_gaussian > -1, surface_sample_ratio_gaussian < 1)
            surface_sample_ratio = torch.where(condition, surface_sample_ratio_gaussian, surface_sample_ratio_uniform)
            #print(surface_sample_ratio)
        else:
            surface_sample_ratio = surface_sample_ratio_uniform
        
        surface_sample_displacement = surface_sample_ratio * surface_sample_range_scaled
        repeated_dist = distances.repeat(surface_sample_n,1)
        surface_sample_dist_ratio = surface_sample_displacement/repeated_dist + 1.0 # 1.0 means on the surface

        surface_repeated_points = shift_points.repeat(surface_sample_n,1)
        if normal_guided_sampling:
            normal_direction = normal_torch.repeat(surface_sample_n,1) # normals are oriented towards sensors.
            #note that normals are oriented towards origin (inwards)
            surface_sample_points = sensor_origin_torch + surface_repeated_points + surface_sample_displacement * (-normal_direction)
        else:
            surface_sample_points = sensor_origin_torch+ surface_repeated_points*surface_sample_dist_ratio

        # only near surface samples are assigned to semantic labels.
        if sem_label_torch is not None:
            surface_sem_label_tensor = sem_label_torch.repeat(1, surface_sample_n).transpose(0,1)
        
        # Part 2. free space uniform sampling
        repeated_dist = distances.repeat(freespace_sample_n,1)
        if sem_label_torch is not None:
            free_sem_label_tensor = torch.zeros_like(repeated_dist)

        free_max_ratio = free_sample_end_dist_scaled / repeated_dist + 1.0
        free_diff_ratio = free_max_ratio - free_min_ratio
        free_sample_dist_ratio = torch.rand(point_num*freespace_sample_n, 1, device=dev)*free_diff_ratio + free_min_ratio
        free_sample_displacement = (free_sample_dist_ratio - 1.0) * repeated_dist
        free_repeated_points = shift_points.repeat(freespace_sample_n,1)
        free_sample_points = free_repeated_points*free_sample_dist_ratio + sensor_origin_torch

        # remove near-surface samples from free-space samples
        tr = surface_sample_range_scaled*1.33
        valid_mask = torch.logical_or(free_sample_displacement < -tr, free_sample_displacement > tr).reshape(-1)
        free_sample_displacement = free_sample_displacement[valid_mask]
        free_sample_points = free_sample_points[valid_mask]

        all_sample_points = torch.cat((surface_sample_points,free_sample_points),0)
        all_sample_displacement = torch.cat((surface_sample_displacement, free_sample_displacement),0)
        
        weight_tensor = torch.ones_like(all_sample_displacement)

        # give a flag indicating the type of the sample [negative: freespace, positive: surface]
        weight_tensor[point_num*surface_sample_n:] *= -1.0
        
        # assign sdf labels to the samples
        # projective distance as the label: behind -, in-front +
        sdf_label_tensor = - all_sample_displacement.squeeze(1)

        # assign the normal label to the samples
        normal_label_tensor = None
        if normal_torch is not None:
            surface_normal = normal_torch.repeat(surface_sample_n,1)
            free_normal = normal_torch.repeat(freespace_sample_n,1)
            free_normal = free_normal[valid_mask]
            normal_label_tensor = torch.cat((surface_normal,free_normal),0)
        
        # assign the semantic label to the samples (including free space as the 0 label)
        sem_label_tensor = None
        if sem_label_torch is not None:
            free_sem_label_tensor = free_sem_label_tensor[valid_mask]
            sem_label_tensor = torch.cat((surface_sem_label_tensor, free_sem_label_tensor),0).int().reshape(-1, 3)
        
        # samples to voxel int coords
        all_sample_voxels = kal.ops.spc.quantize_points(all_sample_points, self.config.tree_level_world)
        all_sample_morton = kal.ops.spc.points_to_morton(all_sample_voxels)

        samples = {}
        samples["count"] = sdf_label_tensor.shape[0]
        #samples["point_morton_count"] = point_morton_count
        samples["pcd_count"] = point_num
        samples["coord"] = all_sample_points.reshape(-1,3)
        samples["voxel_coord"] = all_sample_voxels.reshape(-1, 3)
        samples["morton"] = all_sample_morton.reshape(-1)
        samples["sdf"] = sdf_label_tensor.reshape(-1)
        samples["normal"] = normal_label_tensor.reshape(-1, 3)
        samples["sem"] = sem_label_tensor
        samples["weight"] = weight_tensor.reshape(-1)
        
        return samples