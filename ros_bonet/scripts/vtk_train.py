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
#import rosbonet_cpp_extension as cpp_ext
from os import path as osp
import os
import copy
from os import listdir
import deepdish as dd
from indoor3d_util import room2blocks
import random

"""
    ValueError: Cannot feed value of shape (4, 24, 2, 3) for Tensor u'Y_bbvert:0', which has shape '(?, 100, 2, 3)'

"""

class Data_Configs:
    #sem_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    #             'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    #sem_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12]

    sem_names = ['bg', 'box']
    sem_ids = [0,1]

    points_cc = 9
    sem_num = len(sem_names)
    ins_max_num = 100
    train_pts_num = 4096
    test_pts_num = 4096

class Data_vtk:
    """
    @brief Replacement of Data_S3DIS for vtk_dataset.
    """
    def __init__(self, dataset_dir, usage, train_batch_size=4):
        self.src_usage_dir = osp.join(dataset_dir, 'src', usage)
        self.nframe = -1
        for im_fn in listdir(self.src_usage_dir):
            base, ext = osp.basename(im_fn).split('.')
            index_number, data_name = base.split('_')
            if data_name != 'pointscloud':
                continue
            index_number = int(index_number)
            self.nframe = max(self.nframe, index_number)

        self.nframe += 1
        self.total_train_batch_num = self.nframe
        self.train_next_bat_index=0
        self.train_batch_size = train_batch_size

    @staticmethod
    def get_bbvert_pmask_labels(pc, ins_labels):
        gt_bbvert_padded = np.zeros((Data_Configs.ins_max_num, 2, 3), dtype=np.float32)
        gt_pmask = np.zeros((Data_Configs.ins_max_num, pc.shape[0]), dtype=np.float32)
        count = -1
        unique_ins_labels = np.unique(ins_labels)
        for ins_ind in unique_ins_labels:
            if ins_ind <= -1: continue
            count += 1
            if count >= Data_Configs.ins_max_num: print('ignored! more than max instances:', len(unique_ins_labels)); continue
        
            ins_labels_tp = np.zeros(ins_labels.shape, dtype=np.int8)
            ins_labels_tp[ins_labels == ins_ind] = 1
            ins_labels_tp = np.reshape(ins_labels_tp, [-1])
            gt_pmask[count,:] = ins_labels_tp
        
            ins_labels_tp_ind = np.argwhere(ins_labels_tp == 1)
            ins_labels_tp_ind = np.reshape(ins_labels_tp_ind, [-1])

            ###### bb min_xyz, max_xyz
            pc_xyz_tp = pc[:, 0:3]
            pc_xyz_tp = pc_xyz_tp[ins_labels_tp_ind]
            gt_bbvert_padded[count, 0, 0] = x_min = np.min(pc_xyz_tp[:, 0])
            gt_bbvert_padded[count, 0, 1] = y_min = np.min(pc_xyz_tp[:, 1])
            gt_bbvert_padded[count, 0, 2] = z_min = np.min(pc_xyz_tp[:, 2])
            gt_bbvert_padded[count, 1, 0] = x_max = np.max(pc_xyz_tp[:, 0])
            gt_bbvert_padded[count, 1, 1] = y_max = np.max(pc_xyz_tp[:, 1])
            gt_bbvert_padded[count, 1, 2] = z_max = np.max(pc_xyz_tp[:, 2])

        return gt_bbvert_padded, gt_pmask



    def shuffle_train_files(self, ep):
        # ep : Have no idea that ep is short for what. but it is sued as random seed.
        # It shuffle self.train_files
        self.train_next_bat_index=0
        return

    def load_raw_data(self):
        if hasattr(self, 'blocks') and self.blocks[0].shape[0] >= self.train_next_bat_index:
            delattr(self, 'blocks')
            self.train_next_bat_index = 0

        if not hasattr(self, 'blocks'):
            fn = osp.join(self.src_usage_dir,'%d_pointscloud.h5'%self.train_next_bat_index)
            pointsclouds = dd.io.load(fn)
            xyzrgb = pointsclouds['xyzrgb'].astype(np.double)

            # TODO Denote that room2blocks expect shifted points cloud as below.
            org_xyz = xyzrgb[:,:3].min(axis=0)
            xyzrgb[:,:3] -= org_xyz

            ins_points = pointsclouds['ins_points'].astype(np.double)
            sem_points = pointsclouds['sem_points'].astype(np.double)
            self.blocks = room2blocks(xyzrgb, sem_points, ins_points,
                    num_point=max(xyzrgb.shape[0], 4096),
                    block_size=1., stride=0.5, random_sample=False, sample_num=None, sample_aug=1)

        xyzrgb = self.blocks[0][self.train_next_bat_index,:,:]
        sems   = self.blocks[1][self.train_next_bat_index,:]
        ins    = self.blocks[2][self.train_next_bat_index,:]
        self.train_next_bat_index += 1
        return xyzrgb, sems, ins

    def load_fixed_points(self):
        pc_xyzrgb, sem_labels, ins_labels = self.load_raw_data()

        ### center xy within the block
        min_x = np.min(pc_xyzrgb[:,0]); max_x = np.max(pc_xyzrgb[:,0])
        min_y = np.min(pc_xyzrgb[:,1]); max_y = np.max(pc_xyzrgb[:,1])
        min_z = np.min(pc_xyzrgb[:,2]); max_z = np.max(pc_xyzrgb[:,2])

        ori_xyz = copy.deepcopy(pc_xyzrgb[:, 0:3])  # reserved for final visualization
        use_zero_one_center = True
        if use_zero_one_center:
            pc_xyzrgb[:, 0:1] = (pc_xyzrgb[:, 0:1] - min_x)/ np.maximum((max_x - min_x), 1e-3)
            pc_xyzrgb[:, 1:2] = (pc_xyzrgb[:, 1:2] - min_y)/ np.maximum((max_y - min_y), 1e-3)
            pc_xyzrgb[:, 2:3] = (pc_xyzrgb[:, 2:3] - min_z)/ np.maximum((max_z - min_z), 1e-3)

        pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)

        ########
        sem_labels = sem_labels.reshape([-1])
        ins_labels = ins_labels.reshape([-1])
        bbvert_padded_labels, pmask_padded_labels = Data_vtk.get_bbvert_pmask_labels(pc_xyzrgb, ins_labels)

        psem_onehot_labels = np.zeros((pc_xyzrgb.shape[0], Data_Configs.sem_num), dtype=np.int8)
        for idx, s in enumerate(sem_labels):
            if sem_labels[idx]==-1: continue # invalid points
            sem_idx = Data_Configs.sem_ids.index(s)
            psem_onehot_labels[idx, sem_idx] =1

        return pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels

    def load_train_next_batch(self):
        bat_pc=[]
        bat_sem_labels=[]
        bat_ins_labels=[]
        bat_psem_onehot_labels =[]
        bat_bbvert_padded_labels=[]
        bat_pmask_padded_labels =[]
        #for file in bat_files:
        for i in range(self.train_batch_size):
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = self.load_fixed_points()
            bat_pc.append(pc)
            bat_sem_labels.append(sem_labels)
            bat_ins_labels.append(ins_labels)
            bat_psem_onehot_labels.append(psem_onehot_labels)
            bat_bbvert_padded_labels.append(bbvert_padded_labels)
            bat_pmask_padded_labels.append(pmask_padded_labels)

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        self.train_next_bat_index+=1
        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels

    def load_test_next_batch_random(self):
        return self.load_train_next_batch()

if __name__=='__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use

    configs = Data_Configs()
    net = BoNet(configs = configs)
    net.creat_folders(name='log', re_train=False)
    net.build_graph()

    script_fn = osp.abspath(__file__)
    pkg_dir = str('/').join(script_fn.split('/')[:-2])
    dataset_dir = osp.join(pkg_dir, 'vtk_dataset')
    data = Data_vtk(dataset_dir, 'train')

    #test_areas =['TestArea']
    train(net, data, test_areas = None)


