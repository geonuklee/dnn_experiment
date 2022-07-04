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

class Data_Virtual(Data_ObbDataset):
    def check_cache(self, name):
        cache_dir = osp.join(name,'cached_block')
        if osp.exists(cache_dir):
            return cache_dir
        makedirs(cache_dir)
        indices = []
        sem_label_box = Data_Configs.sem_ids[ Data_Configs.sem_names.index('box') ]
        vnormalize = np.vectorize(xyz_normalize)
       
        all_pc, all_sem, all_ins = [], [], []
        for i, cx in enumerate( [0., 1.] ):
            z0 = 3.
            r = .45
            n = Data_Configs.train_pts_num
            x = np.random.uniform(cx-r,cx+r, n)
            y = np.random.uniform(-r, r, n)
            z = np.random.uniform(z0-r,z0+r, n)
            rgb = np.zeros( (n,3) )
            xyzrgb = np.stack( (x,y,z, ), axis=1)
            xyzrgb = np.concatenate([xyzrgb, rgb], axis=-1)
            sem_points = np.full( (n,), sem_label_box)
            ins_points = np.full( (n,), i+1)
            all_pc.append(xyzrgb)
            all_sem.append(sem_points)
            all_ins.append(ins_points)
        xyzrgb, all_sem, all_ins = [np.concatenate(l,axis=0) for l in (all_pc,all_sem,all_ins) ]

        mins, maxs = np.amin(xyzrgb[:,:3],axis=0), np.amax(xyzrgb[:,:3],axis=0)
        wvec = maxs - mins
        normalized_xyz = np.zeros( (xyzrgb.shape[0],3), dtype=xyzrgb.dtype)
        for k in range(3):
            normalized_xyz[:,k] = vnormalize(xyzrgb[:,k], mins[k], wvec[k])
        xyzrgb = np.concatenate([xyzrgb, normalized_xyz], axis=-1)
        blocks = get_blocks(xyzrgb, all_ins, all_sem, num_points=Data_Configs.train_pts_num)

        i_pick = 0
        fn_frame = osp.join(cache_dir, '%d.pick'%i_pick )
        pick = {}
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
    dataset_dir = osp.join(pkg_dir, 'virtual_dataset')
    data = Data_Virtual(dataset_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use
    configs = Data_Configs()
    net = BoNet(configs = configs)
    net.creat_folders(name='log', re_train=False)
    net.build_graph()
    train(net, data, test_areas = None)
    print("Train is done")

