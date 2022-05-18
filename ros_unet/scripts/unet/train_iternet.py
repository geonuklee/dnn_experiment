#!/usr/bin/python3
#-*- coding:utf-8 -*-

from os import path as osp
import sys
import pickle
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch.nn.functional as F
import glob2
import cv2
import numpy as np
from segment_dataset import ObbDataset
from util import *
from math import floor, ceil


def Split(wh, step, x):
    b, ch, h, w = x.shape
    dim_h, dim_w = 2, 3

    margin = int( (wh-step)/2 )
    nr = ceil( (h-2*margin)/step )
    nc = ceil( (w-2*margin)/step )
    padded_x = torch.zeros([b, ch, nr*step+2*margin, nc*step+2*margin], dtype=x.dtype)

    padded_x[:, :, :h, :w] = x

    # patches.shape = b, ch, nr, nc, w, h
    patches = padded_x.unfold(dim_h,wh,step).unfold(dim_w,wh,step)
    return patches


def Restore(wh, step, org_shape, patches):
    margin = int( (wh-step)/2 )
    b, channels, nr, nc = patches.shape[:-2]

    dst = torch.zeros([b, channels, nr*step+2*margin, nc*step+2*margin],dtype=patches.dtype)
    _, _, h, w = dst.shape
    for r in range(nr):
        for c in range(nc):
            if r > 0:
                dr = margin
            else:
                dr = 0
            if c > 0:
                dc = margin
            else:
                dc = 0
            r0 = r*step+dr
            r1 = min( r0+step+margin-dr, h )
            c0 = c*step+dc
            c1 = min( c0+step+margin-dc, w)
            
            patch = patches[:,:,r,c, dr:-margin, dc:-margin]
            dst[:,:,r0:r1,c0:c1] = patch

    dst = dst[:,:,:org_shape[-2],:org_shape[-1]]
    return dst

def spliter_test():
    dataset = ObbDataset('obb_dataset')
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for data in loader:
        break
    # 1,480,640,3
    rgb = data['rgb']
    # b, h, w, c
    #rgb = rgb[0,:,:,:]
    
    # b,c,h,w
    rgb = rgb.moveaxis(-1,1)

    wh = 200 
    step = 120
    patches = Split(wh,step, rgb)
    dst = Restore(wh,step,rgb.shape,patches)
    cv2.imshow("dst", dst[0,:,:,:].moveaxis(0,-1).numpy())
    cv2.waitKey()

def train():
    dataset = ObbDataset('obb_dataset')
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    spliter = SplitAdapter(w=100, offset=50)
    device = "cuda:0"
    #model = DuNet().to(device)
    #model.train()
    #optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    for data in train_dataloader:
        print(data['idx'])


if __name__ == '__main__':
    spliter_test()

