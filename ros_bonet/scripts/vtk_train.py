#!/usr/bin/python2
#-*- coding:utf-8 -*-

"""!
@file vtk_train.py
@brief Train 3D-Bonet with vtk_dataset to proove perfomance at alinged boxes.

The vtk_dataset generate from gen_vtkscene.py->gen_ponitcloud.py of ros_unet

"""

import numpy as np
import tensorflow as tf
from bonet import BoNet, Plot, Eval_Tools, train
from os import path as osp
import os
import copy
from os import listdir
from indoor3d_util import room2blocks
import random
import glob2
import shutil
from dataset_train import Data_ObbDataset, Data_Configs, get_blocks, xyz_normalize 
from os import makedirs
import pickle
from os import path as osp

class Data_VtkDataset(Data_ObbDataset):
    def check_cache(self, name):
        cache_dir = osp.join(name,'cached_block')
        if osp.exists(cache_dir):
            return cache_dir
        makedirs(cache_dir)
        #pick = { 'xyzrgb':xyzrgb, 'ins_points':ins_points, 'rgb':rgb, 'K':K, 'D':D}
        pick_files = glob2.glob(osp.join(name,'*.pick'),recursive=False)
        indices = []
        sem_label_box = Data_Configs.sem_ids[ Data_Configs.sem_names.index('box') ]
        vnormalize = np.vectorize(xyz_normalize)

        for i_pick, fn in enumerate(pick_files):
            img_idx = osp.basename(fn).split('.')[0]
            img_idx = int(img_idx)
            with open(fn,'rb') as f:
                pick = pickle.load(f)
            xyzrgb, ins_points = pick['xyzrgb'], pick['ins_points']

            mins, maxs = np.amin(xyzrgb[:,:3],axis=0), np.amax(xyzrgb[:,:3],axis=0)
            wvec = maxs - mins
            normalized_xyz = np.zeros( (xyzrgb.shape[0],3), dtype=xyzrgb.dtype)
            for k in range(3):
                normalized_xyz[:,k] = vnormalize(xyzrgb[:,k], mins[k], wvec[k])
            xyzrgb = np.concatenate([xyzrgb, normalized_xyz], axis=-1)

            ins_points -= 1 # Data_S3DIS assign -1 for ins_labels of bg points.
            xyzrgb = xyzrgb[ins_points > -1]
            ins_points = ins_points[ins_points > -1]
            sem_points = np.full_like(ins_points, sem_label_box)

            blocks = get_blocks(xyzrgb, ins_points, sem_points, num_points=Data_Configs.train_pts_num)
            #pc_xyzrgb = blocks[0]
            #for b in range(pc_xyzrgb.shape[0]):
            #    min_coords = np.amin(pc_xyzrgb[b,:,6:9],axis=0)
            #    max_coords = np.amax(pc_xyzrgb[b,:,6:9],axis=0)
            fn_frame = osp.join(cache_dir, '%d.pick'%i_pick )
            with open(fn_frame,'wb') as f_frame:
                pick['blocks'] = blocks
                pickle.dump(pick, f_frame, protocol=2)
            for j in range(blocks[0].shape[0]):
                indices.append( (i_pick, j) )
        fn_info = osp.join(cache_dir, 'info.pick')
        with open(fn_info,'wb') as f:
            pickle.dump(indices, f, protocol=2)
        return cache_dir


if __name__=='__main__':
    script_fn = osp.abspath(__file__)
    pkg_dir = str('/').join(script_fn.split('/')[:-2])
    dataset_dir = osp.join(pkg_dir, 'vtk_dataset')
    data = Data_VtkDataset(dataset_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use
    configs = Data_Configs()
    net = BoNet(configs = configs)
    net.creat_folders(name='log', re_train=False)
    net.build_graph()
    train(net, data, test_areas = None)
    print("Train is done")


