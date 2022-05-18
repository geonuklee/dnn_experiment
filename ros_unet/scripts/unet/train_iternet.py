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
from unet_model import IterNet
from torch import nn, optim

def spliter_test():
    dataset = ObbDataset('obb_dataset_train')
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for data in loader:
        break
    rgb = data['rgb']
    # b,c,h,w
    rgb = rgb.moveaxis(-1,1)

    spliter = SplitAdapter2(wh = 200,step = 120)
    patches = spliter.put(rgb)
    dst = spliter.restore(patches)

    cv2.imshow("dst", dst[0,:,:,:].moveaxis(0,-1).numpy())
    cv2.waitKey()

def train():
    spliter = SplitAdapter2(200, 150)
    device = "cuda:0"
    model = IterNet().to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    dataset = ObbDataset('obb_dataset_train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    checkpoint_fn = 'obb_dataset_train/iternet.pth'
    try:
        checkpoint = torch.load(checkpoint_fn)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_last = checkpoint['epoch']
        print ("Start with previou weight, epoch last = %d" % epoch_last)
    except:
        print ("Start without previou weight")
        epoch_last = 0


    epoch_last = 0
    n_epoch = 10
    niter0, niter = 0, 0
    for epoch in range(epoch_last, n_epoch):  # loop over the dataset multiple times
        for i, data in enumerate(dataloader):
            #rgb, depth, K = data['rgb'], data['depth']
            input_x = data['input_x']
            input_x = spliter.put(input_x).to(device)

            target = data['outline'].unsqueeze(1) # unsqueeze for spliter
            target = spliter.put(target).squeeze(1)
            target = target.long().to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(input_x)
            # pred : b,n(cls),h,w
            # target : b,h,w dtype=np.int64
            loss = model.loss(pred, target)
            loss.backward()
            optimizer.step()

            if i %100 == 0:
                print("epoch [%d/%d], frame[%d/%d] loss = %f" % (epoch,n_epoch,i,len(dataloader), loss.item()) )
        states = {
            'comment': 'input = cv_bedge, cv_wrinkle, cvgrad',
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }
        torch.save(states, 'obb_dataset_train/iternet_%d.pth'%epoch)
        torch.save(states, checkpoint_fn)

if __name__ == '__main__':
    train()


