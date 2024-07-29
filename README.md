This repository represents the official implementation of the paper [N3-Mapping](https://ieeexplore.ieee.org/abstract/document/10518078/):
```
@article{song2024n3,
  title={N $\^{}$\{$3$\}$ $-Mapping: Normal Guided Neural Non-Projective Signed Distance Fields for Large-scale 3D Mapping},
  author={Song, Shuangfu and Zhao, Junqiao and Huang, Kai and Lin, Jiaye and Ye, Chen and Feng, Tiantian},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

## Installation
#### 1. Clone the repository
```
git clone git@github.com:tiev-tongji/N3-Mapping.git
cd N3-Mapping
```
#### 2. Set up conda environment
```
conda create --name n3 python=3.7
conda activate n3
```
#### 3. Install the key requirement kaolin

Kaolin depends on Pytorch (>= 1.8, <= 1.13.1), please install the corresponding Pytorch for your CUDA version (can be checked by ```nvcc --version```). You can find the installation commands [here](https://pytorch.org/get-started/previous-versions/).

For example, for CUDA version >=11.6, you can use:
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Kaolin now supports installation with wheels. For example, to install kaolin 0.13.0 over torch 1.12.1 and cuda 11.6:
```
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu116.html
```

#### 4. Install the other requirements
```
pip install open3d scikit-image wandb tqdm natsort 
```

## Run
Download the dataset the following script:
```
sh ./scripts/download_maicity.sh
```
Other datasets can also be downloaded in the same way:
```
sh ./scripts/download_ncd_example.sh
sh ./scripts/download_neural_rgbd_data.sh
```
The data should follow the kitti odometry format from [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

Therefore if you need to use [Neural RGBD dataset](https://github.com/dazinovic/neural-rgbd-surface-reconstruction), you can convert this dataset to the KITTI format by using for each sequence:
```
sh ./scripts/convert_rgbd_to_kitti_format.sh
```
Now we take the maicity as an example to show how to run the mapping system.
First you need to check the config file such as `./config/maicity/maicity_incre.yaml` and set the correct path like `pc_path`, `pose_path` and `calib_path`. Then use:
 ```
 python run.py config/maicity/maicity_incre.yaml 
 ```

## Evaluation
Please prepare your reconstructed mesh and corresponding ground truth point cloud. Then set the right data path and evaluation set-up in `./eval/evaluator.py`. Now run:
```
python ./eval/evaluator.py
```
## Contact
Feel free to contact me if you have any questions :)
- Song {[1911204@tongji.edu.cn]()}

## Acknowledgment
Our work is mainly built on [SHINE-Mapping](https://github.com/PRBonn/SHINE_mapping). Many thanks to the authors of this excellent work!
We also appreciate the following great open-source works:
- [Voxfield](https://github.com/VIS4ROB-lab/voxfield) (comparison baseline, inspiration)
- [Voxblox](https://github.com/ethz-asl/voxblox) (comparison baseline)
- [NeRF-LOAM](https://github.com/JunyuanDeng/NeRF-LOAM) (comparison baseline)
- [Loc-NDF](https://github.com/PRBonn/LocNDF)(inspiration)

## TODO
Currently our implementation is more of a proof-of-concept and lacks optimization. We are working on improving this. A more efficient voxel-centric mapping design is on the way.
