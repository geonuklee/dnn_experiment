#!/usr/bin/python2
#-*- coding:utf-8 -*-

import numpy as np
from os import path as osp
import os
import copy
from os import listdir, makedirs
#from indoor3d_util import room2blocks
import random
import glob2
import unet_ext
import cv2
from os import makedirs
import pickle
from bonet import Plot
from bonet import Eval_Tools
from bonet import BoNet
import scipy.stats
import matplotlib.pyplot as plt

def get_colors(pc_semins):
    ins_colors = Plot.random_colors(len(np.unique(pc_semins))+1, seed=2)
    semins_labels = np.unique(pc_semins)
    semins_bbox = []
    Y_colors = np.zeros((pc_semins.shape[0], 3))
    for id, semins in enumerate(semins_labels):
        valid_ind = np.argwhere(pc_semins == semins)[:, 0]
        if semins<=-1:
            tp=[0,0,0]
        else:
            tp = ins_colors[id]
        Y_colors[valid_ind] = tp
    return Y_colors

def get_pkg_dir():
    return osp.abspath( osp.join(osp.dirname(__file__),'..') )

def evaluation(net, test_dataset, configs, min_iou=.5):
    # TODO Evaluation.ttest(), Evaluation.evaluation()
    #total_n = test_dataaset.total_train_batch_num

    TP_FP_Total = {}
    for sem_id in configs.sem_ids:
        TP_FP_Total[sem_id] = {}
        TP_FP_Total[sem_id]['TP'] = 0
        TP_FP_Total[sem_id]['FP'] = 0
        TP_FP_Total[sem_id]['Total'] = 0

    for i in range(test_dataset.scene_num):
        bat_pc, bat_sem_gt, bat_ins_gt, bat_psem_onehot, bat_bbvert, bat_pmask, _ \
                = test_dataset.load_next_scene()

        gap = .01 #gap = 5e-3
        volume_num = int(1. / gap) + 2
        volume = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
        volume_sem = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)

        target = [net.y_psem_pred, net.y_bbvert_pred_raw, net.y_bbscore_pred_raw, net.y_pmask_pred_raw]
        feed_dict = {net.X_pc: bat_pc[:, :, 0:9], net.is_train: False}

        [y_psem_pred_sq_raw, y_bbvert_pred_raw, y_bbscore_pred_sq_raw, y_pmask_pred_sq_raw] = \
            net.sess.run(target,feed_dict=feed_dict)

        pc_all = []; ins_gt_all = []; sem_pred_all = []; sem_gt_all = []
        for b in range(y_psem_pred_sq_raw.shape[0]):
            pc = bat_pc[b].astype(np.float16)
            sem_gt = bat_sem_gt[b].astype(np.int16)
            ins_gt = bat_ins_gt[b].astype(np.int32)
            sem_pred_raw = np.asarray(y_psem_pred_sq_raw[b], dtype=np.float16)
            bbvert_pred_raw = np.asarray(y_bbvert_pred_raw[b], dtype=np.float16)
            bbscore_pred_raw = y_bbscore_pred_sq_raw[b].astype(np.float16)
            pmask_pred_raw = y_pmask_pred_sq_raw[b].astype(np.float16)
            sem_pred = np.argmax(sem_pred_raw, axis=-1)
            pmask_pred = pmask_pred_raw * np.tile(bbscore_pred_raw[:, None], [1, pmask_pred_raw.shape[-1]])
            ins_pred = np.argmax(pmask_pred, axis=-2)
            ins_sem_dic = Eval_Tools.get_sem_for_ins(ins_by_pts=ins_pred, sem_by_pts=sem_pred)
            Eval_Tools.BlockMerging(volume, volume_sem, pc[:, :3], ins_pred, ins_sem_dic, gap)
            pc_all.append(pc)
            ins_gt_all.append(ins_gt)
            sem_pred_all.append(sem_pred)
            sem_gt_all.append(sem_gt)
            #print('%d/%d' % (np.unique(ins_pred).shape[0], np.unique(bat_ins_gt).shape[0] ) )

        pc_all = np.concatenate(pc_all, axis=0)
        ins_gt_all = np.concatenate(ins_gt_all, axis=0)
        sem_pred_all = np.concatenate(sem_pred_all, axis=0)
        sem_gt_all = np.concatenate(sem_gt_all, axis=0)

        pc_xyz_int = (pc_all[:, :3] / gap).astype(np.int32)
        ins_pred_all = volume[tuple(pc_xyz_int.T)]

        ###################
        # pred ins
        ins_pred_by_sem = {}
        for sem in configs.sem_ids: ins_pred_by_sem[sem] = []
        ins_idx, cnts = np.unique(ins_pred_all, return_counts=True)
        for ins_id, cn in zip(ins_idx, cnts):
            if ins_id <= -1: continue
            tmp = (ins_pred_all == ins_id)
            sem = scipy.stats.mode(sem_pred_all[tmp])[0][0]
            #if cn <= 0.3*mean_insSize_by_sem[sem]: continue  # remove small instances
            ins_pred_by_sem[sem].append(tmp)
        # gt ins
        ins_gt_by_sem = {}
        for sem in configs.sem_ids: ins_gt_by_sem[sem] = []
        ins_idx, cnts = np.unique(ins_gt_all, return_counts=True)
        for ins_id, cn in zip(ins_idx, cnts):
            if ins_id <= -1: continue
            tmp = (ins_gt_all == ins_id)
            sem = scipy.stats.mode(sem_gt_all[tmp])[0][0]
            #if cn <= 0.3*mean_insSize_by_sem[sem]: continue  # remove small instances
            if len(np.unique(sem_gt_all[ins_gt_all == ins_id])) != 1: print('sem ins label error'); exit()
            ins_gt_by_sem[sem].append(tmp)
        # to associate
        for sem_id, sem_name in zip(configs.sem_ids, configs.sem_names):
            ins_pred_tp = ins_pred_by_sem[sem_id]
            ins_gt_tp = ins_gt_by_sem[sem_id]

            flag_pred = np.zeros(len(ins_pred_tp), dtype=np.int8)
            for i_p, ins_p in enumerate(ins_pred_tp):
                iou_max = -1
                for i_g, ins_g in enumerate(ins_gt_tp):
                    u = ins_g | ins_p
                    i = ins_g & ins_p
                    iou_tp = float(np.sum(i)) / (np.sum(u) + 1e-8)
                    if iou_tp > iou_max:
                        iou_max = iou_tp
                if iou_max >= min_iou:
                    flag_pred[i_p] = 1
            ###
            TP_FP_Total[sem_id]['TP'] += np.sum(flag_pred)
            TP_FP_Total[sem_id]['FP'] += len(flag_pred) - np.sum(flag_pred)
            TP_FP_Total[sem_id]['Total'] += len(ins_gt_tp)
    ###############
    pre_all = []
    rec_all = []
    for sem_id, sem_name in zip(configs.sem_ids, configs.sem_names):
        TP = TP_FP_Total[sem_id]['TP']
        FP = TP_FP_Total[sem_id]['FP']
        Total = TP_FP_Total[sem_id]['Total']
        pre = float(TP) / (TP + FP + 1e-8)
        rec = float(TP) / (Total + 1e-8)
        if Total > 0:
            pre_all.append(pre)
            rec_all.append(rec)
    return pre_all, rec_all

def train(net, train_dataset, valid_dataset, test_dataset, configs):
    # TODO 인식률 올라가는거 확인한다음에 graph 저장 추가
    n_ep = 10000
    valid_aps = []
    test_aps = []
    iters = []
    niter = 0
    fig = plt.figure('train for %s'%train_dataset.name.split('/')[0], figsize=(4,3), dpi=100)
    net.saver.save(net.sess, save_path=net.train_mod_dir + 'model.cptk')
    net.saver.restore(net.sess,net.train_mod_dir+'model.cptk')
    l_min = 0.00001
    for ep in range(0, n_ep,1):
        # TODO 이거 너무 낮은거 아니야?..
        l_rate = max(0.0005/(2**(ep//20)), l_min)
        train_dataset.shuffle_train_files(ep)
        total_train_batch_num = train_dataset.total_train_batch_num
        for i in range(total_train_batch_num):
            bat_pc, _, bat_ins_gt, bat_psem_onehot, bat_bbvert, bat_pmask = train_dataset.load_train_next_batch()
            target = [net.optim, net.psemce_loss, net.bbvert_loss, net.bbvert_loss_l2, net.bbvert_loss_ce, net.bbvert_loss_iou,net.bbscore_loss, net.pmask_loss]
            feed_dict = {net.X_pc:bat_pc[:, :, 0:9], net.Y_bbvert:bat_bbvert, net.Y_pmask:bat_pmask, net.Y_psem:bat_psem_onehot, net.lr:l_rate, net.is_train:True}

            try:
                _, ls_psemce, ls_bbvert_all, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask \
                        = net.sess.run(target, feed_dict=feed_dict)
                #assert(i%15!=0)
            except:
                net.saver.restore(net.sess,net.train_mod_dir+'model.cptk')
                l_min /= 2.
                import pdb; pdb.set_trace()
                continue
            niter += 1

            net.saver.save(net.sess, save_path=net.train_mod_dir+'model.cptk')
            if i%20 == 0:
                min_iou = .7
                valid_dataset.shuffle_train_files(ep)
                test_dataset.shuffle_train_files(ep)
                pred, recall = evaluation(net, test_dataset, configs, min_iou)
                test_aps.append(pred[0])

                pred, recall = evaluation(net, valid_dataset, configs, min_iou)
                iters.append(niter)
                valid_aps.append(pred[0])

                print("for ep %d/%d, batch %d/%d, l_rate=%.2e" % (ep, n_ep, i, total_train_batch_num,l_rate) )
                fig.clf()
                plt.xticks(fontsize=7)
                plt.yticks(fontsize=7)
                plt.plot(iters,valid_aps, 'b-', label='valid set' )
                plt.plot(iters,test_aps, 'g-', label='train set' )
                plt.legend(fontsize=7,loc='lower right')
                plt.ylim(-.1, 1.1)
                fig.axes[0].set_xlabel('iter', fontsize=7)
                fig.axes[0].set_ylabel('AP', fontsize=7)
                fig.canvas.draw()
                plt.show(block=False)
                plt.savefig(net.train_mod_dir+'train_pred.png')

                ###### saving model
                net.saver.save(net.sess, save_path=net.train_mod_dir + 'model' + str(niter).zfill(3) + '.cptk')



