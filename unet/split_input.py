#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
from edge_dataset import EdgeDataset
import torch
from torch.utils.data import DataLoader
import cv2

class SplitAdapter:
    def __init__(self, model=None):
        self.model = model
        self.w = 100
        self.offset = 80

    def put(self, x):
        b, c, h0,w0 = x.shape
        n = 0
        for wmin in range(0, w0, self.offset):
            for hmin in range(0, h0, self.offset):
                n+=1
        output=torch.zeros((b*n, c, self.w, self.w), dtype=x.dtype)
        n = 0
        for ib in range(b):
            for wmin in range(0, w0-self.w+1, self.offset):
                for hmin in range(0, h0-self.w+1, self.offset):
                    output[n,:,:,:] = x[ib,:,hmin:hmin+self.w,wmin:wmin+self.w]
                    n+=1
        return output

if __name__ == '__main__':
    dataset = EdgeDataset('train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    spliter = SplitAdapter()

    for i in range(dataset.__len__() ):
        frame = next(iter(dataloader))
        lap = torch.unsqueeze(frame['lap'],1).float()
        lap = spliter.put(lap)

        rgb = frame['rgb'].moveaxis(3,1)
        rgb = spliter.put(rgb)


        #for k in range(rgb.shape[0]):
        #    src = rgb[k,:,:,:].moveaxis(0,-1).numpy()
        #    cv2.imshow("rgb",src)
        #    c = cv2.waitKey()
        #    if c== ord('q'):
        #        exit(1)

