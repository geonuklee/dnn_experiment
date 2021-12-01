#!/usr/bin/python3
#-*- coding:utf-8 -*-
import numpy as np

import cv2
from os import path as osp

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from torch import nn, optim
#import torch.nn.functional as F
#from torchvision import transforms
from os import listdir

class EdgeDataset(Dataset):
    def __init__(self, useage='train'):
        script_fn = osp.abspath(__file__)
        pkg_dir = str('/').join(script_fn.split('/')[:-3])
        self.useage_path = osp.join(pkg_dir, 'edge_dataset', useage)
        im_list = listdir(self.useage_path)

        self.nframe = 0
        self.types = set()
        for im_fn in im_list:
            base, ext = osp.basename(im_fn).split('.')
            if ext != 'npy':
                continue
            index_number, name = base.split('_')
            index_number = int(index_number)
            self.nframe = max(self.nframe, index_number)
            self.types.add(name)
        self.nframe += 1

    def __len__(self):
        return self.nframe

    def __getitem__(self, idx):
        frame = {}
        try:
            for name in self.types:
                fn = "%d_%s.npy"%(idx, name)
                fn = osp.join(self.useage_path, fn)
                assert osp.exists(fn), "EdgeDataset failed to read the file %s" % fn
                frame[name] = np.load(fn)
        except:
            import pdb;
            pdb.set_trace()

        return frame

