#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
from edge_dataset import EdgeDataset
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
    #parser = argparse.ArgumentParser(description='Description for the program')
    #parser.add_argument( "--verbose", "-v", type=int, nargs='?', help="verbose", default=-1)
    #args = parser.parse_args()

    spliter = SplitAdapter()

    # TODO Iternet에 우선 refinery module만 구현해서 학습.
    device = "cuda:0"
    model = IterNet().to(device)
    model.train()
    dataset = EdgeDataset('train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    print("Sum running loss for %d frame.."%len(dataset) )
    n_epoch = 2
    niter0, niter = 0, 0
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        for i in range(dataset.__len__() ):
            data = next(iter(dataloader))
            optimizer.zero_grad(set_to_none=True)

            lap = torch.unsqueeze(data['lap'],1).float()
            lap = spliter.put(lap).to(device)
            index = data['edge'].long()
            index = spliter.put(index).to(device)

            pred = model(lap)

            loss = model.loss(pred, index)
            loss.backward()
            optimizer.step()

            niter  += lap.shape[0]
            niter0 += lap.shape[0]
            if niter0 > 10000:
                print("[%d/%d,%d/%d] loss = %f" % (epoch,n_epoch,i,dataset.__len__(), loss.item()) )
                niter0 = 0

        torch.save(model.state_dict(), 'edge_dataset/iternet_%d.pth'%epoch)
        torch.save(model.state_dict(), 'edge_dataset/iternet.pth')


