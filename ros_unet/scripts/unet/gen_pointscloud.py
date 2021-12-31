#!/usr/bin/python2
#-*- coding:utf-8 -*-

"""!
@file gen_pointscloud.py
@brief Generate pc_dataset from vtk_dataset

Requires :
    *  OpenCV 3.4 build by source when cpp_ext module cannot find cv symbol

* [x] Load intrinsic, depth rgb, (edge) ground truth
* [x] Component segmentation
* [ ] Save points cloud

"""

from util import colors
from segment_dataset import SegmentDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np

import sys
if sys.version[0] == '2':
    import unet_cpp_extension2 as cpp_ext
else:
    import unet_cpp_extension3 as cpp_ext

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    dataset = SegmentDataset('vtk_dataset','train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    verbose = True

    for i, data in enumerate(dataloader):
        mask = (data['gt'] == 2).squeeze(0)
        mask = mask.numpy().astype(np.uint8)

        rgb = data['rgb'].squeeze(0).numpy()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = data['depth'].squeeze(0).numpy()

        info = data['info']
        K = info['K'].numpy().astype(np.float32)
        D = info['D'].numpy().astype(np.float32)
        retval, labels = cv2.connectedComponents(mask)

        xyzrgb, ins_points = cpp_ext.UnprojectPointscloud(rgb, depth, labels, K, D,
                0.05, 0.01)

        if verbose:
            cv2.imshow("box", mask*255)
            dst = np.zeros((labels.shape[0], labels.shape[1], 3), np.uint8)
            for i in range(retval):
                if i == 0:
                    continue
                for j in range(3):
                    dst[labels==i,j] = colors[i % len(colors)][j]
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            cv2.imshow("labels", dst)

            fig = plt.figure(figsize=(14,7))
            ax1 = plt.subplot(1,2,1, projection='3d')
            ax2 = plt.subplot(1,2,2, projection='3d')
            ax1.axis('equal')
            ax2.axis('equal')

            ax1.scatter(xyzrgb[:,0], xyzrgb[:,1], xyzrgb[:,2],
                    edgecolor=None, c=xyzrgb[:,3:], cmap="BGR", linewidth=0, s=5)

            instance_colors = np.zeros((len(ins_points),3))
            for i in range(retval):
                if i == 0:
                    continue
                for j in range(3):
                    instance_colors[ins_points==i,j] = float(colors[i % len(colors)][j]) / 255.

            valid = ins_points > 0
            ax2.scatter(xyzrgb[valid,0], xyzrgb[valid,1], xyzrgb[valid,2],
                    edgecolor='none', c=instance_colors[valid], linewidth=0, s=5, marker='s')
            c = cv2.waitKey(0)
            plt.show(block=True)
            if c == ord('q'):
                exit(1)

