#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np

import cv2
from os import path as osp

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
#import matplotlib.pyplot as plt
#from unet import *

from edge_dataset import EdgeDataset
from unet_model import EdgeNet
import argparse

def put_frame(model, frame):
    lap = torch.unsqueeze(frame['lap'],1).float().to(device)
    #depth = torch.unsqueeze(frame['depth'],1).float().to(device)
    #rgb = torch.moveaxis(frame['rgb'],3,1).float().to(device)/255.
    #input = torch.cat([lap,rgb],dim=1)
    input = lap
    pred = model(input)
    return pred

def visualize(model, put_frame, dataset, dataloader, device):
    for i in range(dataset.__len__() ):
        data = next(iter(dataloader))
        batch_pred = put_frame(model, data)
        batch_pred = batch_pred.to("cpu").detach()
        batch_lap = torch.unsqueeze(data['lap'],1).float()
        batch_gt = data['edge']
        shape = batch_pred.shape
        b,h,w = shape[0],shape[2],shape[3]
        c = 0
        for k in range(b):
            rgb  = data['rgb'][k,:,:,:].numpy()
            #gray = rgb[0,:,:].numpy()
            gt   = batch_gt[k,:].numpy()
            lap = batch_lap[k, 0].numpy()

            pred = batch_pred[k, 1].numpy()
            pred[pred < 0] = 0
            pred[pred > 1.] =0.999

            cv2.imshow("rgb", rgb)
            cv2.imshow("gt", 255*gt)
            cv2.imshow("lap", 255*(lap < -0.2).astype(np.uint8) ) # For helios
            #cv2.imshow("lap", 255*(lap < -0.03).astype(np.uint8) ) # For Azure
            cv2.imshow("pred", pred)
            #cv2.imshow("bpred", 255*((pred>0.5).astype(np.uint8)))
            c = cv2.waitKey()
            if c == ord('q'):
                break
        if c == ord('q'):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description for the program')
    parser.add_argument( "--verbose", "-v", type=int, nargs='?', help="verbose", default=-1)
    args = parser.parse_args()

    device = "cuda:0"
    model = EdgeNet().to(device)
    if args.verbose is not -1:
        dataset = EdgeDataset('test')
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        device = "cuda:0"
        if args.verbose is None:
            fn = 'edgenet.pth'
        else:
            fn = 'edgenet_%d.pth' % args.verbose
        state_dict = torch.load(fn)
        model.load_state_dict(state_dict)
        model.eval()
        visualize(model, put_frame, dataset, dataloader, device)
        exit(1)

    model.train()
    dataset = EdgeDataset('train')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    c, niter = 0, 0

    print("Sum running loss for %d frame.."%len(dataset) )
    n_epoch = 10
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i in range(dataset.__len__() ):
            data = next(iter(dataloader))
            optimizer.zero_grad(set_to_none=True)
            pred = put_frame(model, data)
            index = data['edge'].long()
            loss = model.loss(pred, index.to(device))
            loss.backward()
            optimizer.step()
            niter +=1
            running_loss += loss.item()
        torch.save(model.state_dict(), 'edgenet_%d.pth'%epoch)
        torch.save(model.state_dict(), 'edgenet.pth')
        print("[%d/%d] loss = %f" % (epoch, n_epoch,running_loss) )



