'''
DeepSoRo Network

author  : Ruoyu Wang
created : 01/23/19 05:22 PM
'''

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import *


def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    #init weights/bias
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i]))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class MLP(nn.Sequential):
    '''Nxdin ->Nxd1->Nxd2->...-> Nxdout'''

    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(MLP, self).__init__(*layers)


class Decoder(nn.Module):

    def __init__(self, mlp_dims, fold_dims, prototype):
        super(Decoder, self).__init__()
        assert(mlp_dims[-1]==fold_dims[0])
        self.mlp = MLP(mlp_dims)
        self.fold = MLP(fold_dims)
        self.prototype = prototype
        self.M = prototype.shape[0]

    def forward(self, codeword):
        c = codeword.unsqueeze(1)                # Bx1xK
        c = c.expand(-1, self.M, -1)             # BxMxK
        
        # expand prototype
        B = c.shape[0]                           # extract batch size      
        g = self.prototype.unsqueeze(0)          # 1xMxD
        g = g.expand(B, -1, -1)                  # BxMxD   
        b = self.mlp(g)                          # BxMxK       
        p = self.fold(b + c)                     # BxMx3
        return p


class DeepSoRoNet(nn.Module):
    
    def __init__(self, mlp_dims, fold_dims, prototype):
        super(DeepSoRoNet, self).__init__()
        assert(fold_dims[0]==512)
        resnet = resnet18()
        self.conv1 = nn.Conv2d(15, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.avgpool = resnet.avgpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.Decoder = Decoder(mlp_dims, fold_dims, prototype)     

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        return x

    def decode(self, codeword):
        p = self.Decoder(codeword)
        return p                      # BxMx3

    def forward(self, x):
        codeword = self.encode(x)
        p = self.decode(codeword)
        return p


def prototype_half_cylinder(grid_dims=[100, 100], cuda=True):
    theta = -torch.arange(0, grid_dims[0], dtype=torch.float32) / grid_dims[0] * np.pi
    y = (torch.arange(0, grid_dims[1], dtype=torch.float32) 
            / grid_dims[1] - 0.5).expand(grid_dims[0], -1).t().reshape(-1)
    x = 0.25 * torch.cos(theta).repeat(grid_dims[1])
    z = 0.25 * torch.sin(theta).repeat(grid_dims[1])
    prototype = torch.stack((x, y, z), 1)   
    if cuda:
        prototype = prototype.cuda()
    return prototype


def deepsoronet_vanilla():
    return DeepSoRoNet([3, 256, 512], [512, 512, 512, 3], prototype_half_cylinder())




