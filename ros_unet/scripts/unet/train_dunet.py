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
    try:
        checkpoint = torch.load("segment_dataset/dunet.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_last = checkpoint['epoch']
        print ("Start with previou weight, epoch last = %d" % epoch_last)
    except:
        print ("Start without previou weight")
        epoch_last = 0


    print("Sum running loss for %d batch.."%len(dataloader) )
    epoch_last = 0
    n_epoch = 10
    niter0, niter = 0, 0
    for epoch in range(epoch_last, n_epoch):  # loop over the dataset multiple times
        for i, data in enumerate(dataloader):
            source = data['source']
            optimizer.zero_grad(set_to_none=True)
            #if source == 'labeled':
            #    input_x = torch.cat((th_edge,grad,rgb), dim=1)
            #    input_x = spliter.put(input_x).to(device)
            #    index = data['gt'].long()
            #    index = spliter.put(index).to(device)
            #    pred = model(input_x)
            #    loss = model.loss(pred, index)
            #    loss.backward()
            #    optimizer.step()
            input_x = data['input']
            input_x = spliter.put(input_x).to(device)
            index = data['gt'].long()
            index = spliter.put(index).to(device)
            pred = model(input_x)
            loss = model.loss(pred, index)
            loss.backward()
            optimizer.step()

            if i %100 == 0:
                print("[%d/%d,%d/%d] loss = %f" % (epoch,n_epoch,i,len(dataloader), loss.item()) )
        states = {
            'comment': 'input = cv_bedge, cv_wrinkle, cvgrad',
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
                }
        torch.save(states, 'segment_dataset/dunet_%d.pth'%epoch)
        torch.save(states, 'segment_dataset/dunet.pth')


