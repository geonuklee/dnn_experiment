#!/usr/bin/python3
#-*- coding:utf-8 -*-

"""!
@file segment_dataset.py
@brief Provides SegmentDataset which parse output of gen_vtkscene.py, gen_labeling.py - segment_dataset, vtk_dataset.

if unet_ext doesn't exists, build it as below. 

```bash
    cd exts
    source run.sh
```

"""

from os import path as osp
from os import makedirs
import cv2
import numpy as np
from os import listdir
from torch.utils.data import Dataset
from torch import Tensor
import shutil
import deepdish as dd

from util import ConvertDepth2input

def CovnertGroundTruthPNG2Array(cv_gt):
    rc = cv_gt.shape[:2]
    cv_gt = cv_gt[:rc[0],:rc[1]]
    #frame[name] = np.load(fn)
    np_gt = np.zeros(rc, dtype=np.uint8)
    # White == Edge
    np_gt[np.logical_and(cv_gt[:,:,2] > 200, cv_gt[:,:,1] > 200)] = 1
    # Red == box for bgr
    np_gt[np.logical_and(cv_gt[:,:,2] > 200, cv_gt[:,:,1] < 50)] = 2
    return np_gt

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

        self.npy_types = ['depth', 'rgb', 'grad']

        cache_dir = osp.join(pkg_dir,name,'cache')
        if not osp.exists(cache_dir):
            makedirs(cache_dir)
        self.cache_usaage_dir = osp.join(cache_dir,usage)

        if not osp.exists(self.cache_usaage_dir):
            print("Generate %s"%self.cache_usaage_dir)

            makedirs(self.cache_usaage_dir)
            fn_form = osp.join(self.cache_usaage_dir,"%d_%s.%s")
            for img_idx in range(self.nframe):
                print("%d/%d" % (img_idx+1, self.nframe) )
                fn_depth = osp.join(pkg_dir,name,'src',usage,'%d_depth.npy'%img_idx)
                fn_rgb = osp.join(pkg_dir,name,'src',usage,'%d_rgb.npy'%img_idx)
                fn_gt = osp.join(pkg_dir,name,'src',usage,'%d_gt.png'%img_idx)
                rgb = np.load(fn_rgb)
                depth = np.load(fn_depth)
                cv_gt = cv2.imread(fn_gt)
                np_gt = CovnertGroundTruthPNG2Array(cv_gt)

                fn_info = osp.join(pkg_dir,name,'src',usage,'%d_caminfo.h5'%img_idx)
                info = dd.io.load(fn_info)
                fx, fy = info['K'][0,0], info['K'][1,1]
                grad, noised_edge, cvlap = ConvertDepth2input(depth, fx, fy)

                if depth.shape[0] != info['height'] or depth.shape[1] != info['width']:
                    import pdb; pdb.set_trace()

                np.save(fn_form%(img_idx,"depth","npy"),depth)
                np.save(fn_form%(img_idx,"rgb","npy"),rgb)
                np.save(fn_form%(img_idx,"grad","npy"),grad)
                shutil.copy(fn_gt, fn_form%(img_idx,"gt","png"))

                fn_pc = osp.join(pkg_dir,name,'src',usage,'%d_pointscloud.h5'%img_idx)
                if osp.exists(fn_pc):
                    shutil.copy(fn_pc, fn_form%(img_idx,"pointscloud","h5"))
                shutil.copy(fn_info, fn_form%(img_idx,"info","h5"))

                fn_edgesimulated = osp.join(pkg_dir,name,'src',usage,'%d_edgedetection.png'%img_idx)
                fn_edgeoutput = fn_form%(img_idx,"edgedetection","png")
                if osp.exists(fn_edgesimulated):
                    shutil.copy(fn_edgesimulated, fn_edgeoutput)
                else:
                    cv2.imwrite(fn_edgeoutput, 255*noised_edge)
                    cv2.imshow("gx", grad[:,:,0])
                    cv2.imshow("bedge", 255*noised_edge)
                    c = cv2.waitKey(1)
                    if c == ord('q'):
                        exit()


    def __len__(self):
        return self.nframe

    def __getitem__(self, idx):
        frame = {}
        for name in self.npy_types:
            fn = "%d_%s.npy"%(idx, name)
            fn = osp.join(self.cache_usaage_dir, fn)
            assert osp.exists(fn), "SegementDataset failed to read the file %s" % fn
            frame[name] = np.load(fn)

        # Check optional types
        for name in ['info.h5', 'pointscloud.h5' ]:
            fn = "%d_%s"%(idx, name)
            fn = osp.join(self.cache_usaage_dir, fn)
            if not osp.exists(fn):
                continue
            k, ext = name.split(".")
            if ext == "h5":
                frame[k] = dd.io.load(fn)
            else:
                frame[k] = np.load(fn)

        fn_gt = "%d_gt.png"%idx
        fn_gt = osp.join(self.cache_usaage_dir, fn_gt)
        assert osp.exists(fn_gt), "SegementDataset failed to read the file %s" % fn_gt
        cv_gt = cv2.imread(fn_gt)
        np_gt = CovnertGroundTruthPNG2Array(cv_gt)
       
        fn_edge = "%d_edgedetection.png"%idx
        fn_edge = osp.join(self.cache_usaage_dir, fn_edge)
        cv_edgedetection = cv2.imread(fn_edge, cv2.IMREAD_GRAYSCALE)
        np_edgedetection = (cv_edgedetection > 0).astype(np.uint8)
        frame['gt'] = np_gt
        frame['idx']  = idx
        frame['edgedetection'] = np_edgedetection
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
            if k in ['info', 'pointscloud']:
                # TODO dict로 제공되는 pointscloud 를 종류별로 분류해 batch 처리.
                continue
            try:
                np_batch = np.stack(l)
                batch[k] = Tensor(np_batch)
            except:
                import pdb; pdb.set_trace()

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

