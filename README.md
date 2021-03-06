# DeepSoRo

Ruoyu Wang, Shiheng Wang, Songyu Du, Erdong Xiao, Wenzhen Yuan, Chen Feng

This repository contains PyTorch implementation associated with the paper:
"[Real-time Soft Body 3D Proprioception via Deep Vision-based Sensing](https://arxiv.org/pdf/1904.03820.pdf)", RA-L/ICRA 2020.


## Abstract
Soft bodies made from flexible and deformable materials are popular in many robotics applications, but their proprioceptive sensing has been a long-standing challenge. In other words, there has hardly been a method to measure and model the high-dimensional 3D shapes of soft bodies with internal sensors. We propose a framework to measure the high-resolution 3D shapes of soft bodies in real-time with embedded cameras. The cameras capture visual patterns inside a soft body, and convolutional neural network (CNN) produces a latent code representing the deformation state, which can then be used to reconstruct the body’s 3D shape using another neural network. We test the framework on various soft bodies, such as a Baymax-shaped toy, a latex balloon, and some soft robot fingers, and achieve real-time computation (≤2.5 ms/frame) for robust shape estimation with high precision (≤1% relative error) and high resolution. We believe the method could be applied to soft robotics and human-robot interaction for proprioceptive shape sensing.
## Results
Top row: Predicted 3D shape Bottom row: Ground truth 3D shape
<p align="center">
<img width="160" src="https://github.com/ai4ce/DeepSoRo/raw/master/docs/images/10_60.gif">
<img width="160" src="https://github.com/ai4ce/DeepSoRo/raw/master/docs/images/5000_5050.gif">
<img width="160" src="https://github.com/ai4ce/DeepSoRo/raw/master/docs/images/150_200.gif">
<img width="160" src="https://github.com/ai4ce/DeepSoRo/raw/master/docs/images/80_130.gif">
<img width="160" src="https://github.com/ai4ce/DeepSoRo/raw/master/docs/images/3900_3950.gif">
</p>

A video demo is provided thourgh [this link](https://youtu.be/kVirop7rf8o).

[![DeeoSoRo Video](http://img.youtube.com/vi/kVirop7rf8o/0.jpg)](http://www.youtube.com/watch?v=kVirop7rf8o "DeeoSoRo Video")

## Code
The code of this project is released on [our GitHub repository](https://github.com/ai4ce/DeepSoRo).

## Requirement

We recommand to use conda to install the required packages.

python>=3.6

pytorch>=1.4

pytorch3d>=0.2.0

open3d>=0.10.0

Please check https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md for pytorch3d installation.


## Insturctions
1. Download the dataset: https://drive.google.com/file/d/1mrsSqivo2GCJ_frP_ehMg5cE5K0Yj48y/view?usp=sharing
2. unzip the BaymaxData.zip.

### Train

python train_baymax.py -d ${PATH_TO_DATASET}/BaymaxData/train -o ${PATH_TO_OUTPUTS}

### Test

python test_baymax.py -d ${PATH_TO_DATASET}/BaymaxData/test -m ${PATH_TO_OUTPUTS}/params/ep_${EPOCH_INDEX}.pth 


## Citation
If you find DeepSoRo useful in your research, please cite:
```BibTex
@article{wang2019real,
  title={Real-time Soft Robot 3D Proprioception via Deep Vision-based Sensing},
  author={Wang, Ruoyu and Wang, Shiheng and Du, Songyu and Xiao, Erdong and Yuan, Wenzhen and Feng, Chen},
  journal={arXiv preprint arXiv:1904.03820},
  year={2019}
}
```

