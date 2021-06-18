import os
import torch
import numpy as np
from dataset import BaymaxDataset
from torch.utils.data import DataLoader
from model import *
import argparse
from pytorch3d.ops import knn_points

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datapath', required=True)
parser.add_argument('-m', '--modelpath', required=True)
args = parser.parse_args()


#intrinsic = open3d.camera.PinholeCameraIntrinsic(width=700, height=700, 
#    fx=601.180, fy=600.982, cx=349.5, cy=349.5)
#extrinsic = np.eye(4)
#camera_parameters = open3d.camera.PinholeCameraParameters()
#camera_parameters.intrinsic = intrinsic
#camera_parameters.extrinsic = extrinsic


#if args.visualization is not None:
#    if not os.path.exists(args.visualization):
#        os.mkdir(args.outpath)


def eval_batch(points_pred, points_gt):
    d_1, _, _ = knn_points(points_pred, points_gt)
    d_2, _, _ = knn_points(points_pred, points_gt)
    err_1, _  = d_1.squeeze(-1).max(dim=1)
    err_2, _ = d_2.squeeze(-1).max(dim=1)
    err = torch.cat((err_1.unsqueeze(-1), err_2.unsqueeze(-1)), dim=-1)
    e, _ = err.max(dim=1)
    return e


def eval(dataset, model):
    err = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    model.cuda()
    for batch in dataloader:
        img, pcd_gt = batch['img'].cuda(), batch['pts'].cuda()
        # TODO: add visualization
        pcd_pred = model(img)    
        e = eval_batch(pcd_pred, pcd_gt)
        err.append(e.detach().cpu())
    err = torch.cat(err).numpy()
    print("max error: %f, mean error: %f, median error: %f unit: m" % (err.max(), err.mean(), np.median(err)))


def main():
    dataset = BaymaxDataset(args.datapath)
    model = deepsoronet_vanilla()
    model.load_state_dict(torch.load(args.modelpath))
    eval(dataset, model)


if __name__ == '__main__':
    main()



