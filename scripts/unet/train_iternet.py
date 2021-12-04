#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
from segment_dataset import SegmentDataset
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms

import cv2
from unet_model import IterNet
from util import *

from math import floor

if __name__ == '__main__':
    spliter = SplitAdapter()

    # TODO Iternet에 우선 refinery module만 구현해서 학습.
    device = "cuda:0"
    model = IterNet().to(device)
    model.train()
    dataset = SegmentDataset('train')

    # Must be shuffle for train
    #Without shuffle, dataloader return only image of index 0~batch_size only.
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    print("Sum running loss for %d frame.."%len(dataset) )
    n_epoch = 50
    niter0, niter = 0, 0
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        for i in range(dataset.__len__() ):
            print("epoch %d/%d, train %d/%d " % (epoch, n_epoch, i, len(dataset) ) )
            data = next(iter(dataloader))
            optimizer.zero_grad(set_to_none=True)

            lap = torch.unsqueeze(data['lap5'],1).float()
            rgb = data['rgb'].float().moveaxis(-1,1)/255
            input_x = torch.cat((lap,rgb),dim=1)
            input_x = spliter.put(input_x).to(device)

            index = data['gt'].long()
            index = spliter.put(index).to(device)
            pred = model(input_x)

            loss = model.loss(pred, index)
            loss.backward()
            optimizer.step()

            niter  += lap.shape[0]
            niter0 += lap.shape[0]
            if niter0 > 10000:
                print("[%d/%d,%d/%d] loss = %f" % (epoch,n_epoch,i,dataset.__len__(), loss.item()) )
                niter0 = 0

        torch.save(model.state_dict(), 'segment_dataset/iternet_%d.pth'%epoch)
        torch.save(model.state_dict(), 'segment_dataset/iternet.pth')


