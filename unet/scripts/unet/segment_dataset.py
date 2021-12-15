#!/usr/bin/python3
#-*- coding:utf-8 -*-

from os import path as osp
import cv2
import numpy as np
from os import listdir
from torch.utils.data import Dataset
from torch import Tensor

class SegmentDataset(Dataset):
    def __init__(self, name, useage):
        script_fn = osp.abspath(__file__)
        pkg_dir = str('/').join(script_fn.split('/')[:-3])
        self.useage_path = osp.join(pkg_dir, name, useage)
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
        frame['idx']  = idx
        return frame

from torch.utils.data import DataLoader

class CombinedDatasetLoader:
    def __init__(self, batch_size):
        self.dataset = {}
        self.dataset['labeled'] = SegmentDataset('segment_dataset','train')
        self.dataset['vtk'] = SegmentDataset('vtk_dataset','train')
        self.list = []
        for name, dataset in self.dataset.items():
            indices = [i for i in range(len(dataset)) ]
            np.random.shuffle(indices)
            length = int( len(dataset) / batch_size )
            for i in range(length):
                partial = indices[batch_size*i:batch_size*(i+1)]
                self.list.append( (name, partial) )
        np.random.shuffle(self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        source, indices = self.list[idx]
        frames = {}
        for i in indices:
            frame = self.dataset[source][i]
            if len(frames) == 0:
                for k in frame.keys():
                    frames[k] = []
            for k, v in frame.items():
                frames[k].append(v)
        batch = {}
        for k, l in frames.items():
            np_batch = np.stack(l)
            batch[k] = Tensor(np_batch)
        batch['source'] = source
        return batch

if __name__ == '__main__':
    # Shuffle two dataset while keep single source for each batch
    dataset_loader = CombinedDatasetLoader(batch_size=2)
    for batch in dataset_loader:
        print(batch['source'])
    print(batch)

