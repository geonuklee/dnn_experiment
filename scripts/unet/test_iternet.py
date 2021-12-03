#!/usr/bin/python3
#-*- coding:utf-8 -*-

import cv2
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from segment_dataset import SegmentDataset
from edge_dataset import EdgeDataset
from unet_model import IterNet
from util import *

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description for the program')
    parser.add_argument( "--verbose", "-v", type=int, nargs='?', help="verbose", default=None)
    args = parser.parse_args()

    device = "cuda:0"
    model = IterNet()
    model.to(device)
    if args.verbose is None:
        fn = 'segment_dataset/iternet.pth'
    else:
        fn = 'segment_dataset/iternet_%d.pth' % args.verbose
    state_dict = torch.load(fn)
    model.load_state_dict(state_dict)

    spliter = SplitAdapter()
    dataset = SegmentDataset('train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, data in enumerate(dataloader):
        lap = data['lap5'].unsqueeze(1).float()
        lap = spliter.put(lap).to(device)
        orgb = data['rgb'][0,:,:,:]

        t0 = time.time()
        pred = model(lap)
        pred = spliter.restore(pred)
        edge_pred = pred.moveaxis(1,3).squeeze(0)[:,:,1].detach().numpy()
        box_pred = pred.moveaxis(1,3).squeeze(0)[:,:,2].detach().numpy()
        edge_bpred = ( edge_pred> 0.95)#.astype(np.uint8)*255
        box_bpred = ( box_pred> 0.9)#.astype(np.uint8)*255
        print("etime=%.2f[ms]", (time.time()-t0) * 1000. )

        dst = np.zeros_like(orgb.numpy())
        dst[edge_bpred,:] = 255
        dst[box_bpred, 2] = 255

        cv2.imshow('rgb', orgb.numpy())
        cv2.imshow('dst', dst)
        #cv2.imshow('edge', edge_bpred)
        #cv2.imshow('box', box_bpred)
        c = cv2.waitKey(0)
        if c == ord('q'):
            exit(1)

