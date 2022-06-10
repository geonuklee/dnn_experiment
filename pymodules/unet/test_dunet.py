#!/usr/bin/python3
#-*- coding:utf-8 -*-

import cv2
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from segment_dataset import SegmentDataset
from edge_dataset import EdgeDataset
from unet_model import DuNet
from util import *

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description for the program')
    parser.add_argument( "--verbose", "-v", type=int, nargs='?', help="verbose", default=None)
    args = parser.parse_args()

    device = "cuda:0"
    model = DuNet()
    model.to(device)
    if args.verbose is None:
        fn = 'segment_dataset/dunet.pth'
    else:
        fn = 'segment_dataset/dunet_%d.pth' % args.verbose
    checkpoint = torch.load(fn)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.eval()  # TODO More poor result with it? why?
    print("test with epoch %d" % checkpoint['epoch'] )

    spliter = SplitAdapter(100,99)
    while True:
        dataset = SegmentDataset('segment_dataset', 'valid')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for i, data in enumerate(dataloader):
            #lap = torch.unsqueeze(data['lap5'],1).float()
            blap = (data['lap5'][:,:,:] < -0.03).unsqueeze(1).float()
            grad = data['grad'].float().moveaxis(-1,1)
            rgb = data['rgb'].float().moveaxis(-1,1)/255

            input_x = torch.cat((blap,grad), dim=1)
            input_x = spliter.put(input_x).to(device)

            orgb = data['rgb'][0,:,:,:]
            t0 = time.time()
            pred = model(input_x)
            pred = spliter.restore(pred)
            pred = pred.detach()
            dst = spliter.pred2dst(pred)
            print("etime=%.2f[ms]" % (float(time.time()-t0) * 1000.) )
        
            cv2.imshow('rgb', orgb.numpy())
            cv2.imshow('blap', 255 * blap[0,0,:,:].numpy().astype(np.uint8) )
            cv2.imshow('dst', dst)
            c = cv2.waitKey(200)
            if c == ord('q'):
                exit(1)

