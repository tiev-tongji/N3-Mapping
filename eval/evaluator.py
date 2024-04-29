import csv
from eval_utils import *
import open3d as o3d

dataset_name = "maicity_01_"
# ground truth point cloud (or mesh) file
# (optional masked by the intersection part of all the compared method)
gt_pcd_path = "xxx/dataset/mai_city/gt_map_pc_mai.ply"

pred_mesh_path = "xx/mai_xx.ply"
method_name = "maixx"

# pred_mesh_path = "xxx/baseline/vdb_fusion_xxx.ply"
# method_name = "vdb_fusion_xxx"

# pred_mesh_path = "xxx/baseline/puma_xxx.ply"
# method_name = "puma_xxx"

# evaluation results output file
base_output_folder = "./eval/eval_results/"
output_csv_path = base_output_folder + dataset_name + method_name + "_eval.csv"

# For MaiCity
down_sample_vox = 0.02
dist_thre = 0.1
truncation_dist_acc = 0.2 
truncation_dist_com = 2.0

# For NCD
# down_sample_vox = 0.02
# dist_thre = 0.2
# truncation_dist_acc = 0.4
# truncation_dist_com = 2.0

# For NRGBD
# down_sample_vox = 0.004
# dist_thre = 0.04 #0.04?
# truncation_dist_acc = 0.08
# truncation_dist_com = 0.4 # 0.5?

# evaluation
eval_metric, error_map = eval_mesh(pred_mesh_path, gt_pcd_path, down_sample_res=down_sample_vox, threshold=dist_thre, 
                        truncation_acc = truncation_dist_acc, truncation_com = truncation_dist_com, gt_bbx_mask_on = True,
                        generate_error_map=False)

print(eval_metric)

if not error_map.is_empty():
    o3d.io.write_point_cloud("./eval/eval_results/" + method_name + ".ply", error_map)
# mesh_error_map = generate_mesh_error_map(pred_mesh_path, gt_pcd_path)
# if not mesh_error_map.is_empty():
#     o3d.io.write_triangle_mesh("./eval/eval_results/" + method_name + "_mesh.ply", mesh_error_map)

evals = [eval_metric]

csv_columns = ['MAE_accuracy (cm)', 'MAE_completeness (cm)', 'Chamfer_L1 (cm)', \
        'Precision [Accuracy] (%)', 'Recall [Completeness] (%)', 'F-score (%)', \
        'Inlier_threshold (m)', 'Outlier_truncation_acc (m)', 'Outlier_truncation_com (m)']

try:
    with open(output_csv_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in evals:
            writer.writerow(data)
except IOError:
    print("I/O error")

