#!/usr/bin/python3
#-*- coding:utf-8 -*-

'''
This script evaluates AP of point cloud which is unprojected from 2D segmentation,
    to compare with the performance of 3D-BoNet.
'''
from unetsegment import Segment2DEdgeBased
from unet.segment_dataset import ObbDataset
from unet.util import *
from bonet_dataset.dataset import Data_VtkDataset, Data_Configs
from unet_ext import UnprojectPointscloud

import torch
from unet.iternet import *
import cv2
from datetime import datetime
from os import path as osp
from os import makedirs
import shutil
import argparse

def resize(rgb, depth, gt_marker, edge, K):
    osize = (depth.shape[1], depth.shape[0])
    dsize = (640,480)
    sK = K.copy()
    for i in range(2):
        sK[i,:] *= dsize[i] / osize[i]
    srgb       = cv2.resize(rgb, dsize, interpolation=cv2.INTER_NEAREST)
    sdepth     = cv2.resize(depth, dsize, interpolation=cv2.INTER_NEAREST)
    sgt_marker = cv2.resize(gt_marker, dsize, interpolation=cv2.INTER_NEAREST)
    sedge = cv2.resize(edge, dsize, interpolation=cv2.INTER_NEAREST)
    return srgb, sdepth, sgt_marker, sedge, sK

def EvaluateAPof2D(pred_marker, gt_marker, min_iou):
    #_, gt_small   = remove_small_instance(gt_marker)
    #_, pred_small = remove_small_instance(pred_marker)

    ins_pred_all = pred_marker.reshape((-1,))
    ins_gt_all = gt_marker.reshape((-1,))
    ins_pred_all -= 1
    ins_gt_all -= 1
    ins_gt_tp = []
    ins_idx, cnts = np.unique(ins_gt_all, return_counts=True)
    for ins_id, cn in zip(ins_idx, cnts):
        if ins_id <= -1: continue
        tmp = (ins_gt_all == ins_id)
        ins_gt_tp.append(tmp)
    ins_pred_tp = []
    ins_idx, cnts = np.unique(ins_pred_all, return_counts=True)
    for ins_id, cn in zip(ins_idx, cnts):
        if ins_id <= -1: continue
        tmp = (ins_pred_all == ins_id)
        ins_pred_tp.append(tmp)

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
    TP = np.sum(flag_pred)
    FP = len(flag_pred) - np.sum(flag_pred)
    Total = len(ins_gt_tp)
    return TP, FP, Total


def Evaluate(dataset_name='vtk_dataset_nooffset'):
    # [*] library 분리 
    # [*] python3 - load vtk_dataset as 'ObbDataset', check point cloud
    # [*] python3 - gt set as, evaluation, test 'AP'
    # [*] python3 - train unet - check segment2d with it ...
    fn = 'weights_vtk/iternet.pth'
    input_ch = 6
    device = "cuda:0"
    checkpoint = torch.load(fn)
    model_name = checkpoint['model_name']
    model = globals()[model_name]()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    segment2d = Segment2DEdgeBased('cam0')

    dataset = Data_VtkDataset(dataset_name+'/train', batch_size=1)
    TP, FP, Total = 0, 0, 0
    for i in range(dataset.scene_num):
        bat_pc, bat_sem_gt, bat_ins_gt, bat_psem_onehot, bat_bbvert, bat_pmask, pick \
                = dataset.load_next_scene()
        rgb, depth, gt_marker, outline, K = pick['rgb'], pick['depth'], pick['mask'], pick['edge'], pick['K']
        rgb, depth, gt_marker, outline, K = resize(rgb, depth, gt_marker, outline, K)
        D = pick['D']
        assert( np.linalg.norm(D) < 1e-5 ) # No support for distorted image by 'Convert2InterInput' yet.
        input_x, grad, hessian, outline0, convex_edge = Convert2InterInput(rgb, depth, K[0,0], K[1,1])
        input_x = torch.Tensor(input_x).unsqueeze(0)
        y1, y2, pred = model(input_x)
        del y1, y2, input_x
        pred = pred.to('cpu')
        pred = model.spliter.restore(pred)
        outline = model.spliter.pred2mask(pred)
        del pred
        marker = segment2d.Process(rgb, depth, outline, convex_edge, float(K[0,0]), float(K[1,1]) )
        _TP, _FP, _Total = EvaluateAPof2D(marker, gt_marker, min_iou=.7)
        TP += _TP
        FP += _FP
        Total += _Total
        print('%d/%d' % (i+1, dataset.scene_num) , end='\r')
        cv2.imshow("rgb", rgb)
        cv2.imshow("gt",   GetColoredLabel(gt_marker+1))
        cv2.imshow("outline", outline*255)
        cv2.imshow("pred", GetColoredLabel(marker))
        c = cv2.waitKey(1)
        if ord('q') == c:
            break
    pre = float(TP) / max(TP + FP,1e-8)
    rec = float(TP) / max(Total,1e-8)
    print('Prediction/recall = %f, %f'%(pre, rec) )
    exit(1)

def Train(dataset_name='vtk_dataset_nooffset'):
    min_iou = .7
    input_ch = 6
    device = "cuda:0"
    model_name = 'BigIterNetInterface'
    model = globals()[model_name]()
    model.to(device)
    output_dir = 'weights_vtk'
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    makedirs(output_dir)
    checkpoint_fn = osp.join(output_dir, 'iternet.pth')
    segment2d = Segment2DEdgeBased('cam0')

    optimizer = model.CreateOptimizer()
    n_epoch = 30
    bound_width = 20

    for model.epoch in range(model.epoch+1, n_epoch):  # loop over the dataset multiple times
        model.train()
        trainset = Data_VtkDataset(dataset_name+'/train', batch_size=1)
        for i in range(trainset.scene_num):
            bat_pc, bat_sem_gt, bat_ins_gt, bat_psem_onehot, bat_bbvert, bat_pmask, pick \
                    = trainset.load_next_scene()
            rgb, depth, gt_marker, outline, K = pick['rgb'], pick['depth'], pick['mask'], pick['edge'], pick['K']
            rgb, depth, gt_marker, outline, K = resize(rgb, depth, gt_marker, outline, K)
            D = pick['D']
            assert( np.linalg.norm(D) < 1e-5 ) # No support for distorted image by 'Convert2InterInput' yet.
            input_x, grad, hessian, outline0, convex_edge = Convert2InterInput(rgb, depth, K[0,0], K[1,1])
            validmask = np.zeros_like(depth)
            validmask[bound_width:-bound_width,bound_width:-bound_width] = 1
            validmask = validmask > 0
            validmask[depth==0.] = 0

            input_x = torch.Tensor(input_x).unsqueeze(0)
            #y1, y2, pred = model(input_x)
            model.iternet.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(input_x)
            outline_dist = cv2.distanceTransform( (outline==0).astype(np.uint8), cv2.DIST_L1, cv2.DIST_MASK_3)
            data = {'validmask':   validmask,
                    'outline':     outline,
                    'outline_dist':outline_dist
                    }
            for k, v in data.items():
                data[k] = torch.from_numpy(v).unsqueeze(0).unsqueeze(0)
            loss = model.ComputeLoss(output, data)
            for v in output:
                del v
            for k, v in data.items():
                del v
            del data
            print("niter %d, frame %d/%d"%(model.niter, i, trainset.scene_num), end='\r')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            validset = Data_VtkDataset(dataset_name+'/test', batch_size=1)
            TP, FP, Total = 0, 0, 0
            for i in range(validset.scene_num):
                bat_pc, bat_sem_gt, bat_ins_gt, bat_psem_onehot, bat_bbvert, bat_pmask, pick \
                        = validset.load_next_scene()
                rgb, depth, gt_marker, outline, K = pick['rgb'], pick['depth'], pick['mask'], pick['edge'], pick['K']
                rgb, depth, gt_marker, outline, K = resize(rgb, depth, gt_marker, outline, K)
                D = pick['D']
                assert( np.linalg.norm(D) < 1e-5 ) # No support for distorted image by 'Convert2InterInput' yet.
                input_x, grad, hessian, outline0, convex_edge = Convert2InterInput(rgb, depth, K[0,0], K[1,1])
                input_x = torch.Tensor(input_x).unsqueeze(0)
                y1, y2, pred = model(input_x)
                del y1, y2, input_x
                pred = pred.to('cpu')
                pred = model.spliter.restore(pred)
                outline = model.spliter.pred2mask(pred)
                del pred
                marker = segment2d.Process(rgb, depth, outline, convex_edge, float(K[0,0]), float(K[1,1]) )
                _TP, _FP, _Total = EvaluateAPof2D(marker, gt_marker, min_iou=.7)
                TP += _TP
                FP += _FP
                Total += _Total
            pre = float(TP) / max(TP + FP, 1e-8)
            rec = float(TP) / max(Total, 1e-8)
            print('Epoch %d/%d : prediction/recall = %f, %f' % (model.epoch+1, n_epoch, pre, rec) )
            states = {
                'model_name' : model.__class__.__name__,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'others' : (model.epoch, model.niter,
                    model.min_valid_loss, model.valid_loss_list)
            }
            with torch.no_grad():
                torch.save(states, checkpoint_fn)
            del states

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type',
                    default='t',
                    choices=['t', 'e'])
    args = parser.parse_args()
    if args.type == 't':
        Train()
    else:
        Evaluate(dataset_name='vtk_dataset_separated')
