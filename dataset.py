'''
DeepSoRo dataset

author  : Ruoyu Wang
created : 06/03/20 01:23 AM
'''

from torch.utils.data import Dataset
import numpy as np
import glog as logger
import cv2
import os
import torch

class BaymaxDataset(Dataset):
    
    def __init__(self, path):
        paths = np.load(path)
        self.pcds = paths[:, 0]
        self.imgs = paths[:, 1]
        self.poses = paths[:, 2]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        pose = np.load(self.poses[idx])
        img = np.load(self.imgs[idx]).astype(np.float32) / 255.0
        pcd = open3d.io.read_point_cloud(self.pcds[idx])
        pts = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors)
        return {'img' : torch.FloatTensor(img).permute(2, 0, 1), 
                'pts': torch.FloatTensor(pts),
                'rgb': torch.FloatTensor(rgb),
                'pose': torch.FloatTensor(pose)}
    
