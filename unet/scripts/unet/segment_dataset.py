#!/usr/bin/python3
#-*- coding:utf-8 -*-

from os import path as osp
import cv2
import numpy as np
from os import listdir
from torch.utils.data import Dataset

class SegmentDataset(Dataset):
    def __init__(self, useage='train'):
        script_fn = osp.abspath(__file__)
        pkg_dir = str('/').join(script_fn.split('/')[:-3])
        self.useage_path = osp.join(pkg_dir, 'segment_dataset', useage)
        im_list = listdir(self.useage_path)

        self.nframe = 0
        self.npy_types = ['depth', 'rgb', 'lap3', 'lap5']
        for im_fn in im_list:
            base, ext = osp.basename(im_fn).split('.')
            if ext != 'npy':
                continue
            index_number, name = base.split('_')
            index_number = int(index_number)
            self.nframe = max(self.nframe, index_number)
        self.nframe += 1

    def __len__(self):
        return self.nframe

    def __getitem__(self, idx):
        frame = {}
        rc = None
        for name in self.npy_types:
            fn = "%d_%s.npy"%(idx, name)
            fn = osp.join(self.useage_path, fn)
            assert osp.exists(fn), "SegementDataset failed to read the file %s" % fn
            frame[name] = np.load(fn)
            if rc is None:
                rc = frame[name].shape[:2]

        fn = "%d_gt.png"%idx
        fn = osp.join(self.useage_path, fn)
        #print("fn= %s"%fn)
        assert osp.exists(fn), "SegementDataset failed to read the file %s" % fn
        cv_gt = cv2.imread(fn)[:rc[0],:rc[1]]
        #frame[name] = np.load(fn)
        np_gt = np.zeros(rc, dtype=np.uint8)
        # White == Edge
        np_gt[np.logical_and(cv_gt[:,:,2] > 200, cv_gt[:,:,1] > 200)] = 1
        # Red == box for bgr
        np_gt[np.logical_and(cv_gt[:,:,2] > 200, cv_gt[:,:,1] < 50)] = 2
        frame['gt'] = np_gt

        return frame

