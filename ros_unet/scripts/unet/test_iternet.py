#!/usr/bin/python3
#-*- coding:utf-8 -*-

import cv2
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from segment_dataset import ObbDataset
from iternet import IterNet
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

    spliter = SplitAdapter(128, 100)
    while True:
        #dataset = ObbDataset('obb_dataset_train',augment=False)
        dataset = ObbDataset('obb_dataset_test',augment=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for i, data in enumerate(dataloader):
            t0 = time.time()
            input_x = data['input_x']
            input_x = spliter.put(input_x).to(device)
            y1, y2, pred = model(input_x)
            del y1, y2, input_x
            pred = pred.detach()
            pred = spliter.restore(pred)

            orgb = data['rgb'][0,:,:,:].numpy()
            dst = spliter.pred2dst(pred, orgb)
            t1 = time.time()
            etime = (t1-t0) *1000.
            dst[-25:,:,:] = 255
            msg = 'Scene %5d/%d,   Etime %.2f[msec]'%(i, len(dataloader), etime)
            cv2.putText(dst, msg, (5,dst.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            cv2.imshow('orgb', orgb)
            cv2.imshow('dst', dst)
            if dataset.augment:
                c = cv2.waitKey(0)
            else:
                c = cv2.waitKey(1)
            if c == ord('q'):
                exit(1)

