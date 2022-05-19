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
import deepdish as dd # TODO delete after replacing to ObbDataset
import glob2
import pickle
import rosbag
import torchvision.transforms.functional as TF

from util import ConvertDepth2input, Convert2InterInput

def CovnertGroundTruthPNG2Array(cv_gt, width, height):
    cv_gt = cv_gt[:height,:width]
    #frame[name] = np.load(fn)
    np_gt = np.zeros((height, width), dtype=np.uint8)
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
        self.src_usage_dir = osp.join(pkg_dir,  name, 'src', usage)
        for im_fn in listdir(self.src_usage_dir):
            base, ext = osp.basename(im_fn).split('.')
            if ext != 'npy':
                continue
            index_number, _ = base.split('_')
            index_number = int(index_number)
            self.nframe = max(self.nframe, index_number)
        self.nframe += 1
        self.npy_types = ['depth', 'rgb']


    def __len__(self):
        return self.nframe

    def __getitem__(self, idx):
        frame = {}
        fn_info  = osp.join(self.src_usage_dir, '%d_caminfo.h5'%idx)
        caminfo = dd.io.load(fn_info)

        fn_gt = osp.join(self.src_usage_dir,'%d_gt.png'%idx)
        cv_gt = cv2.imread(fn_gt)
        np_gt = CovnertGroundTruthPNG2Array(cv_gt,caminfo['width'],caminfo['height'])

        for name in self.npy_types:
            fn = "%d_%s.npy"%(idx, name)
            fn = osp.join(self.src_usage_dir, fn)
            assert osp.exists(fn), "SegementDataset failed to read the file %s" % fn
            frame[name] = np.load(fn)

        depth, fx, fy = frame['depth'], caminfo['K'][0,0], caminfo['K'][1,1]
        results = ConvertDepth2input(depth, fx, fy)

        frame['gt'] = np_gt
        frame['idx']  = idx

        input_x = results[0]

        # gt==box에서 wrinkle 제거.
        np_wrinkle = (results[-1]<1).astype(np.uint8)*255
        distmap = cv2.distanceTransform( np_wrinkle, cv2.DIST_L2, cv2.DIST_MASK_3)
        dist_th = 2 #dist_th = int(0.005 * caminfo['height'])
        flat = distmap>dist_th
        frame['gt'][ np.logical_and( frame['gt']==2, ~flat) ] = 0

        #cv2.imshow("distmap", distmap)
        #cv2.imshow("nface", (frame['gt']==2).astype(np.uint8)*255 )
        #if ord('q') == cv2.waitKey():
        #    exit(1)
        fn_edge = osp.join(self.src_usage_dir, "%d_edgedetection.png"%idx)
        if osp.exists(fn_edge):
            cv_edgedetection = cv2.imread(fn_edge, cv2.IMREAD_GRAYSCALE)
            np_edgedetection = (cv_edgedetection > 0).astype(np.uint8)

            input_x[0,:,:] = np_edgedetection

        frame['input'] = results[0]
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

class ObbDataset(Dataset):
    def __init__(self, name, augment=True):
        self.augment=augment
        script_fn = osp.abspath(__file__)
        pkg_dir = str('/').join(script_fn.split('/')[:-3])
        dataset_dir = osp.join(pkg_dir, name)
        assert osp.exists(dataset_dir)
        pick_files = glob2.glob(osp.join(dataset_dir,'*.pick'),recursive=False)
        camid = 'cam0'
        name_depth = '/%s/helios2/depth/image_raw'%camid

        cache_dir = osp.join(dataset_dir, 'cache')
        if not osp.exists(cache_dir):
            makedirs(cache_dir)

            nframe = 0
            for fn in pick_files:
                with open(fn, 'rb') as f:
                    pick = pickle.load(f, encoding='latin1')
                osize = pick['depth'].shape[1], pick['depth'].shape[0]
                mx,my = cv2.initUndistortRectifyMap(pick['K'],pick['D'],None,pick['newK'],osize,cv2.CV_32F)
                bag = rosbag.Bag(pick['fullfn'])
                for _, depth_msg, _ in bag.read_messages(topics=[name_depth]):
                    depth = np.frombuffer(depth_msg.data, dtype=np.float32)\
                            .reshape(depth_msg.height, depth_msg.width)
                    rect_depth = cv2.remap(depth,mx,my,cv2.INTER_NEAREST)
                    fn_frame = osp.join(cache_dir, 'frame%d.pick'%nframe)
                    with open(fn_frame,'wb') as f_frame:
                        pickle.dump({'depth':rect_depth,'fn_pick':fn}, f_frame, protocol=2)
                    nframe+=1
        self.frame_files = glob2.glob(osp.join(cache_dir, '*.pick'))
        fn = self.frame_files[0]
        self.frame_files = sorted(self.frame_files, key=lambda fn:\
                int(osp.basename(fn).split('frame')[1].split('.')[0]) )

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame_fn = self.frame_files[idx]
        with open(frame_fn, 'rb') as f:
            pick_frame = pickle.load(f, encoding='latin1')

        with open(pick_frame['fn_pick'], 'rb') as f:
            pick = pickle.load(f, encoding='latin1')

        cvgt = cv2.imread(pick['cvgt_fn'])
        rgb, depth = pick['rgb'], pick_frame['depth']
        K = pick['newK'].copy()
        if self.augment:
            # Reesize image height to h0 * fx/fy, before rotaiton augment.
            dsize = (rgb.shape[1], int(rgb.shape[0]*K[0,0]/K[1,1]) )
            K[1,1] = K[0,0]
            rgb,depth,cvgt = [cv2.resize(img, dsize, cv2.INTER_NEAREST) for img in [rgb,depth,cvgt] ]
            rgb,depth,cvgt = [Tensor(img) for img in [rgb,depth,cvgt] ]
            rgb,cvgt = [img.long() for img in [rgb,cvgt] ]
            rgb  = rgb.moveaxis(-1,0)
            cvgt = cvgt.moveaxis(-1,0)
            depth = depth.unsqueeze(0) # b,c,h,w

            choice = np.random.choice(4)
            if choice == 0:
                rgb = TF.gaussian_blur(rgb, 5, np.random.uniform(0., 10.) )
            rgb = TF.adjust_saturation(rgb, np.random.uniform(0.8, 1.2) )
            rgb = TF.adjust_brightness(rgb, np.random.uniform(0.8, 1.2))

            angle = np.random.uniform(-30,30)
            rgb,depth,cvgt = [TF.rotate(img, angle) for img in [rgb,depth,cvgt]]
            rgb  = rgb.moveaxis(0,-1)
            cvgt = cvgt.moveaxis(0,-1)
            depth = depth.squeeze(0) # b,c,h,w
            rgb,depth,cvgt = [img.numpy() for img in [rgb,depth,cvgt] ]
            rgb,cvgt = [ img.astype(np.uint8) for img in [rgb,cvgt] ]

        outline = cvgt==255
        outline = np.logical_and(np.logical_and(outline[:,:,0],outline[:,:,1]),
                outline[:,:,2])
        dist = cv2.distanceTransform( (~outline).astype(np.uint8), cv2.DIST_L1, cv2.DIST_MASK_3)
        outline = dist < 1
        input_x = Convert2InterInput(rgb, depth, K[0,0], K[1,1])[0]
        frame = {'rgb':rgb, 'idx':idx, 'input_x': input_x, 'outline':outline, 'depth':depth }
        return frame

if __name__ == '__main__':
    # Shuffle two dataset while keep single source for each batch
    #dataset_loader = CombinedDatasetLoader(batch_size=2)
    #for batch in dataset_loader:
    #    print(batch['source'])
    #print(batch)
    # Check cache for valid also, 
    #dataset = SegmentDataset('segment_dataset','valid')
    #dataset = SegmentDataset('vtk_dataset','valid')
    #for data in dataset:
    #    print("asdf")
    dataset = ObbDataset('obb_dataset')
    for frame in dataset:
        break

