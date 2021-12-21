#!/usr/bin/python3
#-*- coding:utf-8 -*-

from os import path as osp
from os import makedirs
import cv2
import numpy as np
from os import listdir
from torch.utils.data import Dataset
from torch import Tensor
import shutil

# Build rospkg 'unet' before use it
import sys
if sys.version[0] == '2':
    import unet_cpp_extension2 as cpp_ext
else:
    import unet_cpp_extension3 as cpp_ext

class SegmentDataset(Dataset):
    def __init__(self, name, usage):
        script_fn = osp.abspath(__file__)
        pkg_dir = str('/').join(script_fn.split('/')[:-3])
        assert osp.exists(osp.join(pkg_dir,name))

        self.nframe = 0
        for im_fn in listdir(osp.join(pkg_dir,  name, 'src', usage)):
            base, ext = osp.basename(im_fn).split('.')
            if ext != 'npy':
                continue
            index_number, _ = base.split('_')
            index_number = int(index_number)
            self.nframe = max(self.nframe, index_number)
        self.nframe += 1


        self.npy_types = ['depth', 'rgb', 'lap5', 'grad']

        cache_dir = osp.join(pkg_dir,name,'cache')
        if not osp.exists(cache_dir):
            makedirs(cache_dir)
        self.cache_usaage_dir = osp.join(cache_dir,usage)

        if not osp.exists(self.cache_usaage_dir):
            print("Generate %s"%self.cache_usaage_dir)

            makedirs(self.cache_usaage_dir)
            fn_form = osp.join(self.cache_usaage_dir,"%d_%s.%s")
            for img_idx in range(self.nframe):
                fn_depth = osp.join(pkg_dir,name,'src',usage,'%d_depth.npy'%img_idx)
                fn_rgb = osp.join(pkg_dir,name,'src',usage,'%d_rgb.npy'%img_idx)
                fn_gt = osp.join(pkg_dir,name,'src',usage,'%d_gt.png'%img_idx)
                rgb = np.load(fn_rgb)
                depth = np.load(fn_depth)
                lap5 = cv2.Laplacian(depth, cv2.CV_32FC1, ksize=5)
                grad = cpp_ext.GetGradient(depth, 2)

                np.save(fn_form%(img_idx,"depth","npy"),depth)
                np.save(fn_form%(img_idx,"lap5","npy"),lap5)
                np.save(fn_form%(img_idx,"rgb","npy"),rgb)
                np.save(fn_form%(img_idx,"grad","npy"),grad)
                shutil.copy(fn_gt, fn_form%(img_idx,"gt","png"))


    def __len__(self):
        return self.nframe

    def __getitem__(self, idx):
        frame = {}
        rc = None
        for name in self.npy_types:
            fn = "%d_%s.npy"%(idx, name)
            fn = osp.join(self.cache_usaage_dir, fn)
            assert osp.exists(fn), "SegementDataset failed to read the file %s" % fn
            frame[name] = np.load(fn)
            if rc is None:
                rc = frame[name].shape[:2]

        fn = "%d_gt.png"%idx
        fn = osp.join(self.cache_usaage_dir, fn)
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
    #for batch in dataset_loader:
    #    print(batch['source'])
    #print(batch)
    # Check cache for valid also, 
    _ = SegmentDataset('segment_dataset','valid')
    _ = SegmentDataset('vtk_dataset','valid')

