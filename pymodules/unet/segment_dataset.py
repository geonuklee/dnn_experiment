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
import glob2
import pickle
import rosbag
import torchvision.transforms.functional as TF

from .util import Convert2IterInput
from .gen_obblabeling import ParseMarker
import re

from sys import version_info

def CovnertGroundTruthPNG2Array(cv_gt, width, height):
    cv_gt = cv_gt[:height,:width]
    #frame[name] = np.load(fn)
    np_gt = np.zeros((height, width), dtype=np.uint8)
    # White == Edge
    np_gt[np.logical_and(cv_gt[:,:,2] > 200, cv_gt[:,:,1] > 200)] = 1
    # Red == box for bgr
    np_gt[np.logical_and(cv_gt[:,:,2] > 200, cv_gt[:,:,1] < 50)] = 2
    return np_gt


from torch.utils.data import DataLoader

class ObbDataset(Dataset):
    def __init__(self, name, augment=True, max_frame_per_scene=None):
        self.augment=augment
        dataset_dir = name
        assert osp.exists(dataset_dir)
        pick_files = glob2.glob(osp.join(dataset_dir,'*.pick'),recursive=False)
        camid = 'cam0'
        name_depth = '/%s/helios2/depth/image_raw'%camid
        self.pkg_dir = '/'.join(dataset_dir.split('/')[:-1])
        self.dataset_dir = dataset_dir
        self.cache_dir = osp.join(dataset_dir, 'cache')
        if not osp.exists(self.cache_dir):
            makedirs(self.cache_dir)
            for fn in pick_files:
                if version_info.major == 3:
                    with open(fn, 'rb') as f:
                        pick = pickle.load(f, encoding='latin1')
                else:
                    with open(fn, 'rb') as f:
                        pick = pickle.load(f)
                osize = pick['depth'].shape[1], pick['depth'].shape[0]
                mx,my = cv2.initUndistortRectifyMap(pick['K'],pick['D'],None,pick['newK'],osize,cv2.CV_32F)
                bag = rosbag.Bag( osp.join(self.pkg_dir, pick['rosbag_fn']) )
                basename = osp.splitext( osp.basename(fn) )[0]
                for i, (_, depth_msg, _) in enumerate( bag.read_messages(topics=[name_depth]) ):
                    depth = np.frombuffer(depth_msg.data, dtype=np.float32)\
                            .reshape(depth_msg.height, depth_msg.width)
                    rect_depth = cv2.remap(depth,mx,my,cv2.INTER_NEAREST)
                    fn_frame = osp.join(self.cache_dir, '%s_%d.pick'%(basename, i))
                    with open(fn_frame,'wb') as f_frame:
                        pickle.dump({'depth':rect_depth,'fn_pick':osp.basename(fn)}, f_frame, protocol=2)
        #self.frame_files = glob2.glob(osp.join(self.cache_dir, '*.pick'))
        pattern = re.compile(r"(.*)_(\d+)")
        scenes = {}
        for i, frame_file in enumerate( glob2.glob(osp.join(self.cache_dir, '*.pick')) ):
            basename = osp.splitext(osp.basename(frame_file))[0]
            scene, frame_idx = pattern.findall(basename)[0]
            if not scene in scenes:
                scenes[scene] = []
            if max_frame_per_scene is not None:
                if len(scenes[scene]) >= max_frame_per_scene:
                    continue
            scenes[scene].append(frame_file)
        self.frame_files = []
        for files in scenes.values():
            self.frame_files += files

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame_fn = self.frame_files[idx]
        if version_info.major == 3:
            with open(frame_fn, 'rb') as f:
                    pick_frame = pickle.load(f, encoding='latin1')
            with open(osp.join(self.dataset_dir, pick_frame['fn_pick']), 'rb') as f:
                pick = pickle.load(f, encoding='latin1')
        else:
            with open(frame_fn, 'rb') as f:
                    pick_frame = pickle.load(f)
            with open(osp.join(self.dataset_dir, pick_frame['fn_pick']), 'rb') as f:
                pick = pickle.load(f)

        cvgt = cv2.imread( osp.join(self.pkg_dir,pick['cvgt_fn']) )
        # Remove bacgrkground from input
        outline, convex_edges, marker, _, _, _ = ParseMarker(cvgt)
        rgb, depth = pick['rgb'], pick_frame['depth']
        outline = outline.astype(np.uint8)
        convex_edges = convex_edges.astype(np.uint8)

        K = pick['newK'].copy()
        if self.augment:
            # Reesize image height to h0 * fx/fy, before rotaiton augment.
            dsize = (rgb.shape[1], int(rgb.shape[0]*K[0,0]/K[1,1]) )
            K[1,1] = K[0,0]
            rgb,depth,outline,convex_edges = [cv2.resize(img, dsize, cv2.INTER_NEAREST) for img in [rgb,depth,outline,convex_edges] ]
            marker = cv2.resize(marker.astype(np.float), dsize, cv2.INTER_NEAREST).astype(np.int32)
            rgb,depth,outline,convex_edges,marker = [Tensor(img) for img in [rgb,depth,outline,convex_edges,marker] ]
            rgb,outline,convex_edges,marker = [img.long() for img in [rgb,outline,convex_edges,marker] ]
            outline,convex_edges,marker = [img.unsqueeze(-1) for img in [outline,convex_edges,marker]]

            rgb  = rgb.moveaxis(-1,0)
            outline = outline.moveaxis(-1,0)
            convex_edges = convex_edges.moveaxis(-1,0)
            marker = marker.moveaxis(-1,0)
            depth = depth.unsqueeze(0) # b,c,h,w

            #choice = np.random.choice(4)
            #if choice == 0:
            #    rgb = TF.gaussian_blur(rgb, 5, np.random.uniform(0., 10.) )
            rgb = TF.adjust_saturation(rgb, np.random.uniform(0.8, 1.2) )
            rgb = TF.adjust_brightness(rgb, np.random.uniform(0.8, 1.2))

            #angle = np.random.uniform(-30,30)
            angles = np.arange(-45.,45.,15.)
            angle = np.random.choice(angles,1)[0]
            rgb,depth,outline,convex_edges,marker = [TF.rotate(img, angle) for img in [rgb,depth,outline,convex_edges,marker]]

            rgb  = rgb.moveaxis(0,-1)
            depth = depth.squeeze(0) # b,c,h,w
            rgb,depth,outline,convex_edges,marker = [img.numpy() for img in [rgb,depth,outline,convex_edges,marker] ]
            rgb,outline,convex_edges,marker = [ img.astype(np.uint8) for img in [rgb,outline,convex_edges,marker] ]
        else:
            outline = outline.reshape((1,outline.shape[-2], outline.shape[-1]))
            convex_edges = convex_edges.reshape((1,convex_edges.shape[-2], convex_edges.shape[-1]))
            marker  =  marker.reshape((1,outline.shape[-2], outline.shape[-1]))

        input_x = Convert2IterInput(depth, K[0,0], K[1,1], rgb=None)[0]
        cv_outline = outline.reshape(outline.shape[1:])
        cv_marker = marker.reshape(marker.shape[1:])
        outline_dist = cv2.distanceTransform( (cv_outline==0).astype(np.uint8), cv2.DIST_L1, cv2.DIST_MASK_3)

        nbg = np.logical_or(outline[0,:,:]>0,cv_marker > 0)
        dist = cv2.distanceTransform( (~nbg).astype(np.uint8), cv2.DIST_L1, cv2.DIST_MASK_3)
        validmask = dist < 10.
        #cv2.imshow("valid", validmask.astype(np.uint8)*255)
        #cv2.waitKey()
        validmask = validmask.reshape( (1,validmask.shape[0], validmask.shape[1]) )

        frame = {'rgb':rgb, 'depth':depth, 'idx':idx, 'input_x': input_x, 'outline':outline,
                'convex_edges':convex_edges,
                'outline_dist':outline_dist.reshape( (1,outline_dist.shape[0],outline_dist.shape[1]) ),
                'K':K,
                'validmask':validmask,'marker':cv_marker,
                }
        return frame

if __name__ == '__main__':
    dataset = ObbDataset('obb_dataset_alignedroll',max_frame_per_scene=5)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for j, data in enumerate(dataloader):
        data['rgb']
