#!/usr/bin/python3
#-*- coding:utf-8 -*-

import cv2
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from segment_dataset import ObbDataset
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
        fn = 'obb_dataset_train/iternet.pth'
    else:
        fn = 'obb_dataset_train/iternet_%d.pth' % args.verbose
    checkpoint = torch.load(fn)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # TODO More poor result with it? why?
    print("test with epoch %d" % checkpoint['epoch'] )

    spliter = SplitAdapter2(200, 150)
    while True:
        dataset = ObbDataset('obb_dataset_test')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for i, data in enumerate(dataloader):
            input_x = data['input_x']
            input_x = spliter.put(input_x).to(device)
            orgb = data['rgb'][0,:,:,:]
            pred = model(input_x)
            pred = spliter.restore(pred)
            pred = pred.detach()

            dst = spliter.pred2dst(pred, orgb.numpy())
            cv2.imshow('dst', dst)
            c = cv2.waitKey(0)
            if c == ord('q'):
                exit(1)

