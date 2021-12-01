#!/usr/bin/python3
#-*- coding:utf-8 -*-

import cv2
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from edge_dataset import EdgeDataset
from unet_model import IterNet
from util import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description for the program')
    parser.add_argument( "--verbose", "-v", type=int, nargs='?', help="verbose", default=None)
    args = parser.parse_args()

    device = "cuda:0"
    model = IterNet()
    model.to(device)
    if args.verbose is None:
        fn = 'edge_dataset/iternet.pth'
    else:
        fn = 'edge_dataset/iternet_%d.pth' % args.verbose
    state_dict = torch.load(fn)
    model.load_state_dict(state_dict)

    spliter = SplitAdapter()
    dataset = EdgeDataset('test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i in range(dataset.__len__() ):
        data = next(iter(dataloader))
        lap = data['lap'].unsqueeze(1).float()
        lap = spliter.put(lap).to(device)

        orgb = data['rgb']
        #srgb = spliter.put(torch.moveaxis(orgb,3,1))
        #rgb = spliter.restore(srgb).moveaxis(1,3).squeeze(0)

        #index = data['edge'].long()
        #index = spliter.put(index).to(device)
        pred = model(lap)
        pred = spliter.restore(pred)

        npred = pred.moveaxis(1,3).squeeze(0)[:,:,1].detach().numpy()
        bpred = (npred > 0.5).astype(np.uint8)*255

        #import pdb;pdb.set_trace()

        cv2.imshow('rgb', orgb.squeeze(0).numpy())
        cv2.imshow('pred', bpred)
        #cv2.imshow('rgb_dst', rgb.numpy())
        c = cv2.waitKey()
        if c == ord('q'):
            exit(1)



