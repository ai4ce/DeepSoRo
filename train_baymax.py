'''
DeepSoRo train on baymx dataset

author  : Ruoyu Wang
created : 06/03/20 01:25 PM
'''
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch3d.loss import chamfer_distance
from dataset import BaymaxDataset
from model import *
import argparse
import glog as logger


parser = argparse.ArgumentParser(description='train on baymax dataset')
parser.add_argument('-d', '--datapath', type=str, required=True, help='path to data')
parser.add_argument('-o', '--outpath', type=str, required=True, help='path to save model params')
parser.add_argument('-b', '--batch-size', type=int, default=16, help='batch size')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('-e', '--epochs', type=int, default=500, help='number of epochs')
args = parser.parse_args()


if not os.path.exists(args.outpath):
    os.mkdir(args.outpath)
    os.mkdir(os.path.join(args.outpath, 'params'))


def train(dataset, model, batch_size, lr, epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.cuda()
    model.train()
    for ep in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            opt.zero_grad()
            img, pts = batch['img'].cuda(), batch['pts'].cuda()
            pts_pred = model(img)
            loss, _ = chamfer_distance(pts_pred, pts)
            loss.backward()
            opt.step()
            if batch_idx % 10 == 9:
                logger.info('[%d, %5d] loss: %.6f' %
                    (ep + 1, batch_idx + 1, loss.item()))
        torch.save(model.state_dict(), os.path.join(args.outpath, 'params','ep_%d.pth' % (ep + 1)))    


if __name__ == '__main__':
    dataset = BaymaxDataset(args.datapath)
    model = deepsoronet_vanilla()
    train(dataset, model, args.batch_size, args.learning_rate, args.epochs)
