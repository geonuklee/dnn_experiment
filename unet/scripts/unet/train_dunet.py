#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
from segment_dataset import CombinedDatasetLoader
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms

import cv2
from unet_model import DuNet
from util import *

from math import floor

if __name__ == '__main__':
    spliter = SplitAdapter(w=100, offset=99)

    device = "cuda:0"
    model = DuNet().to(device)
    model.train()
    dataloader = CombinedDatasetLoader(batch_size=2)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    print("Sum running loss for %d batch.."%len(dataloader) )
    epoch_last = 0
    n_epoch = 2 # 50
    niter0, niter = 0, 0
    for epoch in range(epoch_last, n_epoch):  # loop over the dataset multiple times
        for i, data in enumerate(dataloader):
            source = data['source']
            optimizer.zero_grad(set_to_none=True)
            lap = torch.unsqueeze(data['lap5'],1).float()
            rgb = data['rgb'].float().moveaxis(-1,1)/255
            if source == 'labeled':
                input_x = torch.cat((lap,rgb), dim=1)
                input_x = spliter.put(input_x).to(device)
                index = data['gt'].long()
                index = spliter.put(index).to(device)
                pred = model(input_x)
                loss = model.loss(pred, index)
                loss.backward()
                optimizer.step()

            input_x = lap
            input_x = spliter.put(input_x).to(device)
            index = data['gt'].long()
            index[index == 2] = 0
            index = spliter.put(index).to(device)
            pred = model(input_x)
            loss = model.loss(pred, index)
            loss.backward()
            optimizer.step()

            if i %100 == 0:
                print("[%d/%d,%d/%d] loss = %f" % (epoch,n_epoch,i,len(dataloader), loss.item()) )

