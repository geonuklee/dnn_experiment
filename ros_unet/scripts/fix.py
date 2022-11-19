#!/usr/bin/python2
#-*- coding:utf-8 -*-

from os import path as osp
import glob2
import pickle
from unet.util import GetColoredLabel

def fix():
    datasets = glob2.glob('/home/geo/catkin_ws/src/ros_unet/obb_dataset_*')
    for dataset_dir in datasets:
        usage = osp.basename(dataset_dir).split('_')[-1]
        files = glob2.glob(osp.join(dataset_dir, '*.pick'))
        for fn in files:
            with open(fn, 'rb') as f:
                pick = pickle.load(f)
                if 'fullfn' in pick:
                    fullfn = pick.pop('fullfn')
                    pick['rosbag_fn'] = osp.join('rosbag_%s'%usage, osp.basename(fullfn) )
                cvgt_fn0 = pick['cvgt_fn']
                if 'home' in 'home' in cvgt_fn0.split('/')[:2]:
                    cvgt_fn = osp.basename(cvgt_fn0)
                    pick['cvgt_fn'] = osp.join('obb_dataset_%s'%usage, cvgt_fn)

            with open(fn, 'wb') as f:
                pickle.dump(pick, f, protocol=2)

from unet.segment_dataset import *

def fix_dataset():
    dataset = ObbDataset('obb_dataset_alignedroll',max_frame_per_scene=5)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for j, data in enumerate(dataloader):
        print(j, data['rgb'].shape)
    pass

def fix_parsing():
    # TODO plane_marker #27이 사라지는 문제 해결.
    fn = 'obb_dataset_test0523/helios_2022-05-06-20-11-00_cam0.png'
    cv_gt = cv2.imread(fn)
    outline, convex_edges, marker, front_marker, planemarker2vertices, \
            (plane_marker, plane2marker, plane2centers) = ParseMarker(cv_gt)
    cv2.imshow('marker', GetColoredLabel(marker))
    cv2.waitKey()

if __name__ == '__main__':
    #fix_dataset()
    fix_parsing()
