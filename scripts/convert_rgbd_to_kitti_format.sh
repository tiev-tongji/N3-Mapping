#!/bin/bash

root_path=/media/shy/Document/dataset/neural_rgbd
sequence_name=thin_geometry
base_path=${root_path}/${sequence_name}

# For NeuralRGBD dataset, set is_focal_file to True, and already_kitti_format_pose to False
# For Replica dataset,    set is_focal_file to False, and already_kitti_format_pose to True

command="python3 ./dataset/rgbd_to_kitti_format.py
        --output_root ${base_path}_kitti_gt
        --depth_img_folder ${base_path}/depth_gt/
        --rgb_img_folder ${base_path}/images/
        --intrinsic_file ${base_path}/focal.txt
        --pose_file ${base_path}/poses.txt
        --is_focal_file True
        --already_kitti_format_pose False
        --vis_on False
        --max_depth_m 5"

echo "Convert RGBD dataset to KITTI format"
eval $command
echo "done"