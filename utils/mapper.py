import wandb
import numpy as np
from collections import deque
from numpy.linalg import inv, norm
from tqdm import tqdm
import open3d as o3d
import kaolin as kal
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import matplotlib.pyplot as plt

from utils.config import SHINEConfig
from utils.tools import *
from utils.loss import *
from utils.data_sampler import dataSampler
from utils.mesher import Mesher
from utils.visualizer import MapVisualizer, random_color_table
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from dataset.kitti_dataset import KITTIDataset

class Mapper():
    def __init__(self, config: SHINEConfig):

        self.config = config
        self.device = config.device
        self.dtype = config.dtype

        self.run_path = setup_experiment(config)

        # initialize the feature octree
        self.octree = FeatureOctree(config)
        # initialize the mlp decoder
        self.geo_mlp = Decoder(config, is_geo_encoder=True)
        self.sem_mlp = Decoder(config, is_geo_encoder=False)

        # Load the decoder model
        if config.load_model:
            loaded_model = torch.load(config.model_path)
            self.geo_mlp.load_state_dict(loaded_model["geo_decoder"])
            print("Pretrained decoder loaded")
            freeze_model(self.geo_mlp) # fixed the decoder
            if config.semantic_on:
                self.sem_mlp.load_state_dict(loaded_model["sem_decoder"])
                freeze_model(self.sem_mlp) # fixed the decoder
            if 'feature_octree' in loaded_model.keys(): # also load the feature octree  
                self.octree = loaded_model["feature_octree"]
                self.octree.print_detail()

        # dataset
        self.dataset = KITTIDataset(config)

        # sampler
        self.sampler = dataSampler(config)

        # mesh reconstructor
        self.mesher = Mesher(config, self.octree, self.geo_mlp, self.sem_mlp)
        self.mesher.global_transform = inv(self.dataset.begin_pose_inv)
    
        # Non-blocking visualizer
        if config.o3d_vis_on:
            self.vis = MapVisualizer()

        # learnable parameters
        self.geo_mlp_param = list(self.geo_mlp.parameters())
        # learnable sigma for differentiable rendering
        self.sigma_size = torch.nn.Parameter(torch.ones(1, device=self.device)*1.0) 
        # fixed sigma for sdf prediction supervised with BCE loss
        self.sigma_sigmoid = config.logistic_gaussian_ratio*config.sigma_sigmoid_m*config.scale

        # # key scan list
        # self.keyframelist = []
        # # key frames local window
        # self.local_frames_list = []

        self.last_frame_origin = np.zeros(3) # record the last frame origin.

        self.window_size = 10 # frames size
        self.window_traj_gap = 50 # meters
        # samples of each frame
        self.frame_samples_list = deque(maxlen = self.window_size)

        # samples count of each frame
        self.frame_samples_count = deque(maxlen = self.window_size)

        self.coord_pool = torch.empty((0, 3), device=self.device, dtype=self.dtype)
        self.voxel_pool = torch.empty((0, 3), device=self.device, dtype=int)
        self.morton_pool = torch.empty((0), device=self.device, dtype=torch.int64)
        self.sdf_label_pool = torch.empty((0), device=self.device, dtype=self.dtype)
        self.normal_label_pool = torch.empty((0, 3), device=self.device, dtype=self.dtype)
        self.sem_label_pool = torch.empty((0), device=self.device, dtype=int)
        self.weight_pool = torch.empty((0), device=self.device, dtype=self.dtype)

        self.extra_sample_pool_index = [] # for extra local sampling
        self.extra_index_pool = torch.empty((0), dtype=torch.int64)
    # update
    def update_samples_pool(self, samples, min_xyz, max_xyz, use_sliding_window = False):
        
        frame_idx = samples["index"]
        # concatenate new samples with samples pool
        self.coord_pool = torch.cat((self.coord_pool, samples["coord"]), 0)
        self.sdf_label_pool = torch.cat((self.sdf_label_pool, samples["sdf"]), 0)
        self.weight_pool = torch.cat((self.weight_pool, samples["weight"]), 0)
        self.voxel_pool = torch.cat((self.voxel_pool, samples["voxel_coord"]), 0)
        self.morton_pool = torch.cat((self.morton_pool, samples["morton"]), 0)
        if samples["normal"] is not None:
            self.normal_label_pool = torch.cat((self.normal_label_pool, samples["normal"]), 0)
        else:
            self.normal_label_pool = None
        if samples["sem"] is not None:
            self.sem_label_pool = torch.cat((self.sem_label_pool, samples["sem"]), 0)
        else:
            self.sem_label_pool = None
        
        if use_sliding_window: 
            maskx = torch.logical_and(self.voxel_pool[:,0] > min_xyz[0],
                                      self.voxel_pool[:,0] < max_xyz[0])
            
            masky = torch.logical_and(self.voxel_pool[:,1] > min_xyz[1],
                                      self.voxel_pool[:,1] < max_xyz[1])
            
            maskz = torch.logical_and(self.voxel_pool[:,2] > min_xyz[2],
                                      self.voxel_pool[:,2] < max_xyz[2])
            
            mask = torch.logical_and(torch.logical_and(maskx, masky),maskz)

            self.coord_pool = self.coord_pool[mask]
            self.sdf_label_pool = self.sdf_label_pool[mask]
            self.weight_pool = self.weight_pool[mask]
            self.voxel_pool = self.voxel_pool[mask]
            self.morton_pool = self.morton_pool[mask]
            if self.normal_label_pool is not None:
                self.normal_label_pool = self.normal_label_pool[mask]
            if self.sem_label_pool is not None:
                self.sem_label_pool = self.sem_label_pool[mask]

        # downsampling samples or extra local sampling
        # pay more attention to observation-less region.
        if self.config.extra_training:
            if frame_idx > 10:
                dups_in_mortons = dict(list_duplicates(self.morton_pool.cpu().numpy().tolist()))
                extra_sample_pool_index = []
                for m, indexes in dups_in_mortons.items():
                    if len(indexes) < 10: # 10
                        extra_sample_pool_index.extend(indexes)
                self.extra_index_pool = torch.tensor(extra_sample_pool_index, device=self.config.device, dtype=torch.int64)

    # TODO: better do it in data sampling
    # filter out all free samples out of voxels.
    def filter_samples(self, samples):
        #t1 = get_time()
        
        pcd_count = samples["pcd_count"]
        surface_sample_num = self.config.surface_sample_n * pcd_count
        #free_sample_num = self.config.free_sample_n * pcd_count
        
        # unpack samples
        count = samples["count"]
        coord = samples["coord"]
        voxel_coord = samples["voxel_coord"]
        morton = samples["morton"]
        sdf = samples["sdf"]
        normal = samples["normal"]
        sem = samples["sem"]
        weight = samples["weight"]

        valid_index = list(range(surface_sample_num))
        voxel_morton = morton.cpu().numpy().tolist() # nodes at certain level

        for idx in range(surface_sample_num, len(voxel_morton)):
            if voxel_morton[idx] in self.octree.nodes_lookup_tables[self.config.tree_level_world]:
                valid_index.append(idx) # nodes to corner dictionary: key is the morton code
       
        samples["coord"] = coord[valid_index]
        samples["voxel_coord"] = voxel_coord[valid_index]
        samples["morton"] = morton[valid_index]
        samples["sdf"] = sdf[valid_index]
        samples["weight"] = weight[valid_index]
        if sem is not None:
            samples["sem"] = sem[valid_index]
        if normal is not None:
            samples["normal"] = normal[valid_index]
        
        samples["count"] = len(valid_index)

        #t2 = get_time()
        #print("filter {:d} samples cost {:.1f}ms".format(count-len(valid_index),1000*(t2-t1)))

        return samples

    # sampling training pairs from label pool.
    def get_batch(self):
        train_sample_count = self.sdf_label_pool.shape[0]
        if not self.config.extra_training:
            index = torch.randint(0, train_sample_count, (self.config.bs,), device=self.config.device)
        else:
            extra_batch_size = round(self.config.bs/3)
            bacth_size = self.config.bs - extra_batch_size
            #bacth_size = self.config.bs
            extra_sample_count = self.extra_index_pool.shape[0]
            index = torch.randint(0, train_sample_count, (bacth_size,), device=self.config.device)
            if extra_sample_count > bacth_size:
                extra_index_index = torch.randint(0, extra_sample_count, (extra_batch_size,), device=self.config.device)
                extra_index = self.extra_index_pool[extra_index_index]
                index = torch.cat((index, extra_index), dim=0)

        coord = self.coord_pool[index, :]
        sdf_label = self.sdf_label_pool[index]
        weight = self.weight_pool[index]
            
        if self.normal_label_pool is not None:
            normal_label = self.normal_label_pool[index, :]
        else: 
            normal_label = None

        if self.sem_label_pool is not None:
            sem_label = self.sem_label_pool[index]
        else: 
            sem_label = None

        return coord, sdf_label, normal_label, sem_label, weight

    def mapping(self):
        processed_frame = 0
        total_iter = 0

        for frame_id in tqdm(range(self.dataset.total_pc_count)):
            if (frame_id < self.config.begin_frame or frame_id > self.config.end_frame or \
                frame_id % self.config.every_frame != 0): 
                continue

            # # for kitti 07, skip the pose jump
            # if (frame_id > 659 and frame_id < 730):
            #     processed_frame += 1
            #     continue

            vis_mesh = False 
            if processed_frame == self.config.freeze_after_frame: # freeze the decoder after certain frame
                print("Freeze the decoder")
                freeze_model( self.geo_mlp) # fixed the decoder

            T0 = get_time()
            # sampling
            _, frame_origin_torch, frame_pc_s_torch, frame_normal_torch, frame_label_torch = self.dataset[frame_id]
            samples = self.sampler.sampling(frame_pc_s_torch, 
                                            frame_origin_torch, 
                                            frame_normal_torch, 
                                            frame_label_torch, 
                                            normal_guided_sampling=self.config.normal_sampling_on)
            # samples = self.sampler.sampling_rectified_sdf(frame_pc_s_torch, 
            #                                 frame_origin_torch, 
            #                                 frame_normal_torch, 
            #                                 frame_label_torch)
            
            # avoid duplicated samples due to the slow motion
            samples["index"] = frame_id
            current_frame_origin = frame_origin_torch.cpu().numpy().reshape(-1)
            relative_dist = np.mean(norm(current_frame_origin - self.last_frame_origin))
            if processed_frame > 5 and relative_dist < 0.5*self.config.scale and self.config.use_keyframe: 
                print("slow motion! jump frame")
                with open('./log/jump.txt', 'a') as f:
                    f.write(f"{frame_id}, {relative_dist/self.config.scale}\n")
                processed_frame += 1
                if (frame_id+1) % self.config.mesh_freq_frame != 0:
                    continue
            self.last_frame_origin = current_frame_origin

            # update feature octree
            if self.config.octree_from_surface_samples:
                # update with the sampled surface points
                self.octree.update(samples["coord"][samples["weight"] > 0, :])
            else:
                # update with the original points
                self.octree.update(frame_pc_s_torch.to("cuda"))
            
            # calculate local boundary
            frame_origin_voxel = kal.ops.spc.quantize_points(frame_origin_torch, self.config.tree_level_world)
            radius_vox_count = round(self.config.pc_radius/self.config.leaf_vox_size)
            min_xyz = frame_origin_voxel[:3] - radius_vox_count #lower bound
            max_xyz = frame_origin_voxel[:3] + radius_vox_count # upper bound

            # update samples pool
            samples = self.filter_samples(samples)
            self.update_samples_pool(samples, min_xyz, max_xyz, use_sliding_window = self.config.sliding_window_on)
            
            octree_feat = list(self.octree.parameters())
            opt = setup_optimizer(self.config, octree_feat, self.geo_mlp_param, None, self.sigma_size)
            self.octree.print_detail()

            T1 = get_time()
            for iter in tqdm(range(self.config.iters)):
                # load batch data (avoid using dataloader because the data are already in gpu, memory vs speed)
                coord, sdf_label, normal_label, sem_label, weight = self.get_batch()
                if self.config.normal_loss_on or self.config.ekional_loss_on:
                    coord.requires_grad_(True)

                # interpolate and concat the hierachical grid features
                # predict the scaled sdf with the feature
                if self.config.predict_residual_sdf:
                    feature, coarse_features = self.octree.query_split_feature(coord)
                    sdf_pred = self.geo_mlp.sum_sdf(feature, coarse_features)
                else:
                    feature = self.octree.query_feature(coord)
                    sdf_pred = self.geo_mlp.sdf(feature)

                if self.config.semantic_on:
                    sem_pred = self.sem_mlp.sem_label_prob(feature)

                # calculate the gradients
                if self.config.normal_loss_on or self.config.ekional_loss_on:
                    g = get_gradient(coord, sdf_pred)*self.sigma_sigmoid
                    g_normalized = F.normalize(g, p=2, dim=1)

                # calculate the loss
                surface_mask = weight > 0

                cur_loss = 0.
                weight = torch.abs(weight) # weight's sign indicate the sample is around the surface or in the free space
                sdf_loss = sdf_bce_loss(sdf_pred, sdf_label, self.sigma_sigmoid, weight, self.config.loss_weight_on, self.config.loss_reduction) 
                cur_loss += sdf_loss

                # incremental learning regularization loss (useless in this work)
                reg_loss = 0.
                if self.config.continual_learning_reg:
                    reg_loss = self.octree.cal_regularization()
                    cur_loss += self.config.lambda_forget * reg_loss

                # optional ekional loss
                eikonal_loss = 0.
                if self.config.ekional_loss_on:
                    #eikonal_loss = ((g[~surface_mask].norm(2, dim=-1) - 1.0) ** 2).mean() # MSE with regards to 1
                    eikonal_loss = ((g.norm(2, dim=-1) - 1.0) ** 2).mean() # MSE with regards to 1  
                    cur_loss += self.config.weight_e * eikonal_loss
                
                normal_loss = 0.
                if self.config.normal_loss_on:
                    normal_loss = normal_diff_loss(g_normalized, normal_label, surface_mask)
                    cur_loss += self.config.weight_n * normal_loss
                
                # semantic classification loss
                # sem_loss = 0.
                if self.config.semantic_on:
                    loss_nll = nn.NLLLoss(reduction='mean')
                    sem_loss = loss_nll(sem_pred[::self.config.sem_label_decimation,:], sem_label[::self.config.sem_label_decimation])
                    cur_loss += self.config.weight_s * sem_loss

                opt.zero_grad(set_to_none=True)
                cur_loss.backward() # this is the slowest part (about 10x the forward time)
                opt.step()

                total_iter += 1

            T2 = get_time()
            
            # reconstruction by marching cubes
            if frame_id == 0 or (processed_frame) % self.config.mesh_freq_frame == 0:
                print("Begin mesh reconstruction from the implicit map")
                vis_mesh = True                
                mesh_path = self.run_path + '/mesh/mesh_frame_' + str(frame_id+1) + ".ply"
                map_path = self.run_path + '/map/sdf_map_frame_' + str(frame_id+1) + ".ply"
                if self.config.mc_with_octree: # default
                    cur_mesh = self.mesher.recon_octree_mesh(self.config.mc_query_level, self.dataset.map_down_pc, self.config.mc_res_m, 
                                                             mesh_path, map_path, self.config.save_map, self.config.semantic_on, 
                                                             filter_free_space_vertices=self.config.clean_mesh_on)
                else:
                    cur_mesh = self.mesher.recon_bbx_mesh(self.dataset.map_bbx, self.dataset.map_down_pc, self.config.mc_res_m, 
                                                          mesh_path, map_path, self.config.save_map,self. config.semantic_on,
                                                          filter_free_space_vertices=self.config.clean_mesh_on)
                # save raw point cloud
                # pc_map_path = self.run_path + '/map/pc_frame_' + str(frame_id+1) + ".ply"
                # self.dataset.write_merged_pc(pc_map_path)
            
            T3 = get_time()

            if self.config.o3d_vis_on:
                if vis_mesh: 
                    cur_mesh.transform(self.dataset.begin_pose_inv) # back to the globally shifted frame for vis
                    self.vis.update(self.dataset.cur_frame_pc, self.dataset.cur_pose_ref, cur_mesh)
                else: # only show frame and current point cloud
                    self.vis.update(self.dataset.cur_frame_pc, self.dataset.cur_pose_ref)

            processed_frame += 1
        
        print("Begin mesh reconstruction from the implicit map")
        mesh_path = self.run_path + '/mesh/final_mesh.ply'
        map_path = self.run_path + '/map/final_sdf.ply'
        if self.config.mc_with_octree: # default
            cur_mesh = self.mesher.recon_octree_mesh(self.config.mc_query_level, self.dataset.map_down_pc, self.config.mc_res_m, 
                                                        mesh_path, map_path, self.config.save_map, self.config.semantic_on, 
                                                        filter_free_space_vertices=self.config.clean_mesh_on)
        else:
            cur_mesh = self.mesher.recon_bbx_mesh(self.dataset.map_bbx, self.dataset.map_down_pc, self.config.mc_res_m, 
                                                    mesh_path, map_path, self.config.save_map,self. config.semantic_on,
                                                    filter_free_space_vertices=self.config.clean_mesh_on)
        if self.config.o3d_vis_on:
            cur_mesh.transform(self.dataset.begin_pose_inv) # back to the globally shifted frame for vis
            self.vis.update(self.dataset.cur_frame_pc, self.dataset.cur_pose_ref, cur_mesh)
            self.vis.stop()
           

