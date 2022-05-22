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
#from unet_model import IterNet
from iternet import IterNet, get_w_from_pixel_distribution, weighted_bce_loss
from torch import nn, optim
from datetime import datetime


def spliter_test():
    dataset = ObbDataset('obb_dataset_train')
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for data in loader:
        break
    rgb = data['rgb']
    # b,c,h,w
    rgb = rgb.moveaxis(-1,1)

    spliter = SplitAdapter(wh = 200,step = 120)
    patches = spliter.put(rgb)
    dst = spliter.restore(patches)

    cv2.imshow("dst", dst[0,:,:,:].moveaxis(0,-1).numpy())
    cv2.waitKey()

def train():
    spliter = SplitAdapter(128, 100)
    device = "cuda:0"
    model = IterNet().to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    dataset = ObbDataset('obb_dataset_train',augment=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

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
    niter = 0
    for epoch in range(epoch_last, n_epoch):  # loop over the dataset multiple times
        for i, data in enumerate(dataloader):
            input_x = data['input_x']
            input_x = spliter.put(input_x).to(device)

            target = data['outline']
            target = spliter.put(target).float().to(device)
            optimizer.zero_grad(set_to_none=True)

            y1, y2, y3 = model(input_x)
            w1, w2 = get_w_from_pixel_distribution(target)

            loss1 = weighted_bce_loss(y1, target, w1, w2)
            loss2 = weighted_bce_loss(y2, target, w1, w2)
            loss3 = weighted_bce_loss(y3, target, w1, w2)
            lambda1, lambda2, lambda3 = 1e-1, 2e-1, 3e-1
            loss  = lambda1*loss1 + lambda2*loss2 + lambda3*loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            states = {
                'comment': 'input = cv_bedge, cv_wrinkle, cvgrad',
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }

            if i %100 == 0 and niter > 0:
                current_time = datetime.now().strftime("%H:%M:%S")
                print("epoch [%d/%d], frame[%d/%d] loss = %f" \
                        % (epoch,n_epoch,i,len(dataloader), loss.item()),
                        current_time )
                torch.save(states, 'obb_dataset_train/iternet_%d.pth'%epoch)
                torch.save(states, checkpoint_fn)
            niter += 1
        torch.save(states, 'obb_dataset_train/iternet_%d.pth'%epoch)
        torch.save(states, checkpoint_fn)


if __name__ == '__main__':
    train()
