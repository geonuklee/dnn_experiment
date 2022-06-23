#!/usr/bin/python2
#-*- coding:utf-8 -*-

import numpy as np
from os import path as osp
import os
import copy
from os import listdir, makedirs
import deepdish as dd
import glob2
import unet_ext
import cv2
from os import makedirs
import pickle

from dataset_train import *
from unet.segment_dataset import ObbDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from bonet import Eval_Tools

def test2():
    pkg_dir = get_pkg_dir()
    dataset_dir ='/home/docker/obb_dataset_train'
    data = Data_ObbDataset(dataset_dir, batch_size=4)
    for i in range(data.total_train_batch_num):
        bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels,\
                bat_bbvert_padded_labels, bat_pmask_padded_labels\
                    = data.load_train_next_batch()
        print(i, bat_pc.shape)
    print("test is done")
    return


def test1():
    pkg_dir = get_pkg_dir()
    dataset_dir ='/home/docker/obb_dataset_train'
    dataset = ObbDataset(dataset_dir, False, max_frame_per_scene=1)
    sem_label_box = Data_Configs.sem_ids[ Data_Configs.sem_names.index('box') ]

    for data in dataset:
        bgr, depth, marker = data['rgb'], data['depth'], data['marker']
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        K = data['K']
        D = np.zeros((4,1),dtype=K.dtype)
        xyzrgb, ins_points\
                = unet_ext.UnprojectPointscloud(rgb,depth,marker,K,D,leaf_xy=0.02,leaf_z=0.01)
        ins_points -= 1 # Data_S3DIS assign -1 for ins_labels of bg points.
        sem_points = np.full_like(ins_points, -1)
        sem_points[ins_points > -1] = sem_label_box

        xyzrgb = xyzrgb[ins_points>-1,:]
        ins_points = ins_points[ins_points > -1]
        sem_points = np.full_like(ins_points, sem_label_box)

        in_range = xyzrgb[:,2] < 2.
        xyzrgb     = xyzrgb[in_range]
        ins_points = ins_points[in_range]
        sem_points = sem_points[in_range]

        fig = plt.figure(1, figsize=(12,8), dpi=100)
        fig.clf()

        sub_rows, sub_cols = 2, 4

        ax = fig.add_subplot(sub_rows, sub_cols, 1)
        ax.scatter(xyzrgb[:,0]/xyzrgb[:,2],-xyzrgb[:,1]/xyzrgb[:,2], c=xyzrgb[:,3:6], s=4, linewidths=0)
        ax.set_title('img plane')
        ax.axis('equal')

        ax = fig.add_subplot(sub_rows, sub_cols, 2, projection='3d')
        ax.scatter(xyzrgb[:,0], xyzrgb[:,1], c=xyzrgb[:,3:6], s=4, linewidths=0)
        ax.view_init(elev=-90, azim=-90)
        ax.set_title('org cloud')
        ax.axis('off')
        ax.axis('equal')

        blocks = get_blocks(xyzrgb, sem_points, ins_points, num_points=4000)
        ax = fig.add_subplot(sub_rows, sub_cols, 3, projection='3d')
        ax.set_title('overlap block cloud')
        for ch_idx in range(blocks[0].shape[0]):
            pc   = blocks[0][ch_idx,:,:3]
            rgb  = blocks[0][ch_idx,:, 3:]
            rgb  = blocks[0][ch_idx,:,3:]
            sems = blocks[1][ch_idx,:]
            ins  = blocks[2][ch_idx,:]
            ax.scatter(pc[:,0],pc[:,1],pc[:,2], c=rgb, s=4, linewidths=0)
        ax.view_init(elev=-90, azim=-90)
        ax.axis('off')
        ax.axis('equal')

        #### center xy within the block
        pc_xyzrgb = blocks[0]
        min_x = np.min(pc_xyzrgb[:,:,0]); max_x = np.max(pc_xyzrgb[:,:,0])
        min_y = np.min(pc_xyzrgb[:,:,1]); max_y = np.max(pc_xyzrgb[:,:,1])
        min_z = np.min(pc_xyzrgb[:,:,2]); max_z = np.max(pc_xyzrgb[:,:,2])
        ori_xyz = copy.deepcopy(pc_xyzrgb[:,:,0:3])  # reserved for final visualization
        pc_xyzrgb[:,:,0:1] = (pc_xyzrgb[:,:,0:1] - min_x)/ np.maximum((max_x - min_x), 1e-3)
        pc_xyzrgb[:,:,1:2] = (pc_xyzrgb[:,:,1:2] - min_y)/ np.maximum((max_y - min_y), 1e-3)
        pc_xyzrgb[:,:,2:3] = (pc_xyzrgb[:,:,2:3] - min_z)/ np.maximum((max_z - min_z), 1e-3)
        pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)

        ax = fig.add_subplot(sub_rows, sub_cols, 4, projection='3d')
        ax.set_title('normalized cloud')
        for ch_idx in range(blocks[0].shape[0]):
            pc   = pc_xyzrgb[ch_idx,:,:3]
            rgb  = blocks[0][ch_idx,:, 3:]
            sems = blocks[1][ch_idx,:]
            ins  = blocks[2][ch_idx,:]
            #print(np.amin(pc, axis=0), np.amax(pc, axis=0) )
            ax.scatter(pc[:,0],pc[:,1],pc[:,2], c=rgb, s=4, linewidths=0)
        ax.set_zticks([])
        ax.view_init(elev=-90, azim=-90)
        ax.axis('equal')

        '''
        # TODO 
        * [x] Visualize ins label
        * [x] Simulate Unsynced label << !!!!
        * [x] Test Eval_Tools.BlockMerging
        '''
        ax = fig.add_subplot(sub_rows, sub_cols, 5, projection='3d')
        ax.set_title('Org ins label')
        colors = get_colors(np.array(range(100)))
        for ch_idx in range(blocks[0].shape[0]):
            pc   = pc_xyzrgb[ch_idx,:,:3]
            sems = blocks[1][ch_idx,:]
            ins  = blocks[2][ch_idx,:]
            ax.scatter(pc[:,0],pc[:,1],pc[:,2], c=colors[ins], s=4, linewidths=0)
        ax.set_zticks([])
        ax.view_init(elev=-90, azim=-90)
        ax.axis('equal')
        ax.axis('off')
        ax.axis('equal')

        ax = fig.add_subplot(sub_rows, sub_cols, 6, projection='3d')
        ax.set_title('Simulated ins label')
        print('unique of gt ins = ', np.unique(blocks[2]) )
        block_sem_pred = blocks[1].copy()
        block_ins_pred = np.zeros_like(blocks[2])
        for ch_idx in range(blocks[0].shape[0]):
            pc   = pc_xyzrgb[ch_idx,:,:3]
            sems = blocks[1][ch_idx,:]
            ins  = blocks[2][ch_idx,:]
            new_ins =ins + 10 *ch_idx
            block_ins_pred[ch_idx,:] = new_ins
            ax.scatter(pc[:,0],pc[:,1],pc[:,2], c=colors[new_ins], s=4, linewidths=0)
        ax.set_zticks([])
        ax.view_init(elev=-90, azim=-90)
        ax.axis('equal')
        ax.axis('off')
        ax.axis('equal')

        gap = .01
        volume_num = int(1. / gap) + 2
        volume = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
        volume_sem = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
        pc_all = []
        for batch in range(pc_xyzrgb.shape[0]):
            ins_pred = block_ins_pred[batch]
            sem_pred = block_sem_pred[batch]
            ins_sem_dic = Eval_Tools.get_sem_for_ins(ins_by_pts=ins_pred, sem_by_pts=sem_pred)
            Eval_Tools.BlockMerging(volume, volume_sem, pc_xyzrgb[batch,:,:3], ins_pred, ins_sem_dic, gap)
            #Eval_Tools.BlockMerging(volume, volume_sem, pc_xyzrgb[batch,:,-3:], ins_pred, ins_sem_dic, gap)
            pc_all.append(pc_xyzrgb[batch,:,:])
        pc_all = np.concatenate(pc_all, axis=0)
        pc_xyz_int = (pc_all[:,:3] / gap).astype(np.int32)
        #pc_xyz_int = (pc_all[:,-3:] / gap).astype(np.int32)
        ins_pred_all = volume[tuple(pc_xyz_int.T)]

        ax = fig.add_subplot(sub_rows, sub_cols, 7, projection='3d')
        ax.set_title('Merging ins label')
        ax.scatter(pc_all[:,0],pc_all[:,1],pc_all[:,2], c=colors[ins_pred_all], s=4, linewidths=0)
        ax.set_zticks([])
        ax.view_init(elev=-90, azim=-90)
        ax.axis('off')
        ax.axis('equal')
        
        plt.axis('tight')
        plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
        plt.show(block=True)
        break

if __name__ == '__main__':
    test1()
