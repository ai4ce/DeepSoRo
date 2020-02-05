# DeepSoRo

Ruoyu Wang, Shiheng Wang, Songyu Du, Erdong Xiao, Wenzhen Yuan, Chen Feng

This repository contains PyTorch implementation associated with the paper:
"[Real-time Soft Body 3D Proprioception via Deep Vision-based Sensing](https://arxiv.org/pdf/1904.03820.pdf)",


## Abstract
Soft bodies made from flexible and deformable materials are popular in many robotics applications, but their proprioceptive sensing has been a long-standing challenge. In other words, there has hardly been a method to measure and model the high-dimensional 3D shapes of soft bodies with internal sensors. We propose a framework to measure the high-resolution 3D shapes of soft bodies in real-time with embedded cameras. The cameras capture visual patterns inside a soft body, and convolutional neural network (CNN) produces a latent code representing the deformation state, which can then be used to reconstruct the body’s 3D shape using another neural network. We test the framework on various soft bodies, such as a Baymax-shaped toy, a latex balloon, and some soft robot fingers, and achieve real-time computation (≤2.5 ms/frame) for robust shape estimation with high precision (≤1% relative error) and high resolution. We believe the method could be applied to soft robotics and human-robot interaction for proprioceptive shape sensing. Our code is available at: https://ai4ce.github.io/DeepSoRo/.
## Results
Top row: Predicted 3D shape Bottom row: Ground truth 3D shape
<p align="center">
<img width="160" src="./docs/images/10_60.gif">
<img width="160" src="./docs/images/5000_5050.gif">
<img width="160" src="./docs/images/150_200.gif">
<img width="160" src="./docs/images/80_130.gif">
<img width="160" src="./docs/images/3900_3950.gif">
</p>

## Code
The code of this project will be released soon.

## Citation
If you find DeepSoRo useful in your research, please cite:
```BibTex
@article{wang2019real,
  title={Real-time Soft Robot 3D Proprioception via Deep Vision-based Sensing},
  author={Wang, Ruoyu and Wang, Shiheng and Xiao, Erdong and Jindal, Kshitij and Yuan, Wenzhen and Feng, Chen},
  journal={arXiv preprint arXiv:1904.03820},
  year={2019}
}
```
