# PTT: PointTrackTransformer

## Overview

- [Introduction](#introduction)
- [Performance](#performance)
- [Setup](#setup)
- [QuickStart](#quickstart)
- [Acknowledgment](#acknowledgment)
- [Citation](#citation)

## Introduction

This is the official code release of "**PTT: Point-Track-Transformer Module for 3D Single Object Trackingin Point Clouds**"(**Accepted** as Contributed paper in **[IROS 2021](https://www.iros2021.org/)**).  :star2: :star2: :star2: 

[conference paper](https://ieeexplore.ieee.org/document/9636821)  |    [video(youtube)](https://youtu.be/Cajj6iHFvrc)    |    [video(bilibili)](https://www.bilibili.com/video/av291947183)

<p align="center">
<img src="docs/sot.gif" width="800"/>
</p>

  This work is towards the point-based 3D SOT (**S**ingle **O**bject **T**racking) task, and is dedicated to solving several challenges brought by the natural **sparsity** of point cloud, such as: ***error accumulation***, ***sparsity sensitivity***, and ***feature ambiguity***. 

To this end, we proposed our PTT, a framework combining transformer and tracking pipeline. The main pipeline of PTT is as following. Experiments show that tracker can well achieve robust tracking in sparse point cloud scenes (less than 50 foreground points) by using Transformer's Self Attention to re-weight sparse features.

<img src="docs/pipeline.png" alt="main-pipeline"  />

## Performance

Here, we show the latest performance of our PTT. In order to better open source our code, we reconstruct the code and optimized some parameters compared to the version in the paper. **It is worth noting that** we unified the environment and parameter settings of the final version, so the model performance is slightly different from the paper. The performances after code reconstruction are as follows:

### kitti dataset

|           | Car  | Ped  | Cyclist | Van  |
| --------- | ---- | ---- | ------- | ---- |
| Success   | 69.0 | 47.7 | 41.0    | 55.3 |
| Precision | 82.1 | 72.2 | 49.4    | 64.0 |

### nuScenes dataset

|           | Car  | Truck | Bus  | Trailer |
| --------- | ---- | ----- | ---- | ------- |
| Success   | 40.2 | 46.5  | 39.4 | 51.7    |
| Precision | 45.8 | 46.7  | 36.7 | 46.5    |

*For nuScenes, we follow the settings of [BAT](https://github.com/Ghostish/Open3DSOT) to retrain and test our model. And these results are all trained with batchsize 48 on a single Nvidia RTX 3090, while the results of [extended journal paper](https://ieeexplore.ieee.org/document/9695195) are trained with 8 x 2080Ti GPUs.*

## Setup

### installation

0. install some dependences

   ```
   apt update && apt-get install git libgl1 -y
   ```

1. create conda env and install python 3.8

   ```bash
   conda create -n ptt python=3.8 -y
   conda activate ptt
   git clone https://github.com/shanjiayao/PTT
   cd PTT/
   ```

2. install torch

   ```bash
   pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   ```

   *It is worth noting that we tested our code on different versions of **cuda**, and finally found that the performance will be different due to the randomness of the cuda version. **So please use cuda version at least 11.0 and install torch follow the above command.***

3. install others

   ```bash
   pip install -r requirements.txt
   conda install protobuf -y
   ```

4. [*optional*] install visualize tools

   ```bash
   pip install vtk==9.0.1
   pip install mayavi==4.7.4 pyqt5==5.15.6
   ```

5. setup ptt package

   ```bash
   python setup.py develop   # ensure be root dir
   ```

### dataset configuration

1. Kitti

   Download the dataset from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and organize the downloaded files as follows:

   ```bash
   PTT                                           
   |-- data                                     
   |   |-- kitti                                                                          
   │   │   └── training
   │   │       ├── calib
   │   │       ├── label_02
   │   │       └── velodyne
   
   ```

2. nuScenes

   Download the dataset from [nuScenes](https://www.nuscenes.org/) and organize the downloaded files as follows:

   ```bash
   PTT                                           
   |-- data              
   |   └── nuScenes                                                      
   |       |── maps
   |       |── samples
   |       |── sweeps
   |       └── v1.0-trainval
   ```

## QuickStart


### configs

The model configs are located within [tools/cfgs](https://github.com/open-mmlab/OpenPCDet/blob/master/tools/cfgs) for different datasets. Please refer to [ptt.yaml](tools/cfgs/kitti_models/ptt.yaml) to learn more introduction about the model configs.

### pretrained models

Here we provide the **pretrained models** on both kitti and nuscenes dataset. You can download these models from [google drive](https://drive.google.com/drive/folders/1k7bVP087GzV16ysMAdiGefZvJoXVAAtL?usp=sharing). Then organize the downloaded files as follows:

```
PTT
├── output
│   ├── kitti_models
│   └── nuscenes_models
```

### train

For training, you can customize the training by modifying the parameters in the yaml file of the corresponding model, such as '**CLASS_NAMES**', '**OPTIMIZATION**', '**TRAIN**' and '**TEST**'.

After configuring the yaml file, run the following command to parser the path of config file and the training tag.

```bash
cd PTT/tools
# python train_tracking.py --cfg_file cfgs/kitti_models/ptt.yaml --extra_tag car
python train_tracking.py --cfg_file $model_config_path --extra_tag $your_train_tag
```

*By default, we use a single Nvidia RTX 3090 for training.*

For training with ddp, you can execute the following command ( ensure be root dir ):

```bash
# bash scripts/train_ddp.sh 2 --cfg_file cfgs/kitti_models/ptt.yaml --extra_tag car
bash scripts/train_ddp.sh $NUM_GPUs --cfg_file $model_config_path --extra_tag $your_train_tag
```

### eval

Similar to training, you need to configure parameters such as '**CLASS_NAMES**' in the yaml file first, and then run the following commands to test single checkpoint.

```bash
cd PTT/tools
# python test_tracking.py --cfg_file cfgs/kitti_models/ptt.yaml --extra_tag car --ckpt ../output/kitti_models/ptt/car/ckpt/best_model.pth
python test_tracking.py --cfg_file $model_config_path --extra_tag $your_train_tag --ckpt $your_saved_ckpt
```

If you need to test all models, you could modify the default value of '**eval_all**' in [here](https://github.com/shanjiayao/PTT/blob/master/tools/test_tracking.py#L36) before running above command.

After evaluation, the results are saved to the same path as the model, such as 'output/kitti_models/ptt/car/'.

## Acknowledgment

- This repo is built upon [P2B](https://github.com/HaozheQi/P2B) and [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
- Thank [Ghostish](https://github.com/Ghostish) for his implementation of [BAT](https://github.com/Ghostish/Open3DSOT). 
- Thank [qq456cvb](https://github.com/qq456cvb) for his implementation of [Point-Transformers](https://github.com/qq456cvb/Point-Transformers). 

## Citation

If you find the project useful for your research, you may cite,

```
@INPROCEEDINGS{ptt,
  author={Shan, Jiayao and Zhou, Sifan and Fang, Zheng and Cui, Yubo},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={PTT: Point-Track-Transformer Module for 3D Single Object Tracking in Point Clouds}, 
  year={2021},
  volume={},
  number={},
  pages={1310-1316},
  doi={10.1109/IROS51168.2021.9636821}}
```

```
@ARTICLE{ptt-journal,
  author={Jiayao, Shan and Zhou, Sifan and Cui, Yubo and Fang, Zheng},
  journal={IEEE Transactions on Multimedia}, 
  title={Real-time 3D Single Object Tracking with Transformer}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2022.3146714}}
```

