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

import torch
from unet.iternet import *
import cv2
from datetime import datetime
from os import path as osp
from os import makedirs
import shutil
import argparse
import pickle

threshold_curvature = 10.
min_iou = .7

def resize(rgb, depth, gt_marker, edge, K):
    osize = (depth.shape[1], depth.shape[0])
    dsize = (640,480)
    sK = K.copy()
    for i in range(2):
        sK[i,:] *= float(dsize[i]) / float(osize[i])
    srgb       = cv2.resize(rgb, dsize, interpolation=cv2.INTER_NEAREST)
    sdepth     = cv2.resize(depth, dsize, interpolation=cv2.INTER_NEAREST)
    sgt_marker = cv2.resize(gt_marker, dsize, interpolation=cv2.INTER_NEAREST)
    sedge = cv2.resize(edge, dsize, interpolation=cv2.INTER_NEAREST)
    return srgb, sdepth, sgt_marker, sedge, sK

def EvaluateAPof2D(depth, outline, pred_marker, gt_marker, min_iou):
    bg_markers = pred_marker.copy()
    uniq_marker, counts = np.unique(bg_markers[depth<.001],return_counts=True)
    if counts.size > 0:
        bg = uniq_marker[np.argmax(counts)]
        pred_marker[pred_marker==bg] = -1
    pred_marker[pred_marker==0] = -1
    pred_marker[depth < .001] = -1
    #_, outliers = remove_small_instance(pred_marker,20)
    #pred_marker[pred_marker==outliers] = -1
    min_counts = 50

    gt_marker[gt_marker==0] = -1

    ins_pred_all = pred_marker.reshape((-1,)).copy()
    ins_gt_all = gt_marker.reshape((-1,)).copy()

    ins_gt_tp = []
    ins_gt_idx, cnts = np.unique(ins_gt_all, return_counts=True)
    for ins_id, cn in zip(ins_gt_idx, cnts):
        if ins_id <= -1: continue
        #if cn < min_counts: # glitch
        #    continue
        tmp = (ins_gt_all == ins_id)
        ins_gt_tp.append(tmp)

    ins_pred_tp = []
    ins_pred_idx, cnts = np.unique(ins_pred_all, return_counts=True)
    for ins_id, cn in zip(ins_pred_idx, cnts):
        if ins_id <= -1: continue
        if cn < min_counts: # glitch which is supposed to be filtered by OBB process
            continue
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
        input_x, grad, hessian, outline0, convex_edge = Convert2InterInput(rgb, depth, K[0,0], K[1,1],threshold_curvature)
        input_x = torch.Tensor(input_x).unsqueeze(0)
        y1, y2, pred = model(input_x)
        del y1, y2, input_x
        pred = pred.to('cpu')
        pred = model.spliter.restore(pred)
        pred_outline = model.spliter.pred2mask(pred)
        del pred
        marker = segment2d.Process(rgb, depth, pred_outline, convex_edge, float(K[0,0]), float(K[1,1]) )
        _TP, _FP, _Total = EvaluateAPof2D(depth, pred_outline, marker, gt_marker, min_iou)
        TP += _TP
        FP += _FP
        Total += _Total
        print('%d/%d - TP %d, FP %d, Total %d' % (i+1, dataset.scene_num, _TP, _FP, _Total) , end='\r')
        cv2.imshow("rgb", rgb)
        cv2.imshow("gt",   GetColoredLabel(gt_marker+1))
        cv2.imshow("pred", GetColoredLabel(marker+1))
        cv2.imshow("gt_outline", outline*255)
        cv2.imshow("pred_outline", pred_outline*255)
        cv2.imshow("pred_outline0", 255*(outline0>0).astype(np.uint8) )
        if _TP < _Total or _FP > 0:
            c = cv2.waitKey(1)
        else:
            c = cv2.waitKey(1)
        if ord('q') == c:
            break
    pre = float(TP) / max(TP + FP,1e-8)
    rec = float(TP) / max(Total,1e-8)
    print('Prediction/recall = %f, %f'%(pre, rec) )
    exit(1)

def Train(dataset_name='vtk_dataset_nooffset'):
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
    n_epoch = 100
    bound_width = 20
    n_converge = 0
    break_condition = False

    log_pick = {'valid_ap':[], 'valid_ar':[], 'train_ap':[], 'train_ar':[], 'iter':[], 'ep':[] }
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
            input_x, grad, hessian, outline0, convex_edge = Convert2InterInput(rgb, depth, K[0,0], K[1,1],threshold_curvature=.1)

            #if i==0:
            #    print(pick['fn'])
            #    cv2.imshow("gt_outline", 255*(outline>0).astype(np.uint8) )
            #    cv2.imshow("pred_outline0", 255*(outline0>0).astype(np.uint8) )
            #    cv2.waitKey(1)

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
            model.niter += 1
            print("niter %d, frame %d/%d"%(model.niter, i, trainset.scene_num), end='\r')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            trainset2 = Data_VtkDataset(dataset_name+'/train', batch_size=1)
            validset = Data_VtkDataset(dataset_name+'/test', batch_size=1)
            for name, dataset in {'train':trainset2, 'valid':validset}.items():
                TP, FP, Total = 0, 0, 0
                for i in range(dataset.scene_num):
                    bat_pc, bat_sem_gt, bat_ins_gt, bat_psem_onehot, bat_bbvert, bat_pmask, pick \
                            = dataset.load_next_scene()
                    rgb, depth, gt_marker, outline, K = pick['rgb'], pick['depth'], pick['mask'], pick['edge'], pick['K']
                    rgb, depth, gt_marker, outline, K = resize(rgb, depth, gt_marker, outline, K)
                    D = pick['D']
                    assert( np.linalg.norm(D) < 1e-5 ) # No support for distorted image by 'Convert2InterInput' yet.
                    input_x, grad, hessian, outline0, convex_edge = Convert2InterInput(rgb, depth, K[0,0], K[1,1], threshold_curvature)
                    input_x = torch.Tensor(input_x).unsqueeze(0)
                    y1, y2, pred = model(input_x)
                    del y1, y2, input_x
                    pred = pred.to('cpu')
                    pred = model.spliter.restore(pred)
                    pred_outline = model.spliter.pred2mask(pred)
                    del pred
                    if i==0:
                        cv2.imshow("gt_outline", 255*(outline>0).astype(np.uint8) )
                        cv2.imshow("pred_outline0", 255*(outline0>0).astype(np.uint8) )
                        cv2.imshow("pred_outline", 255*(pred_outline>0).astype(np.uint8) )
                        cv2.waitKey(1)
                    outline[depth<.001] = 1
                    marker = segment2d.Process(rgb, depth, pred_outline, convex_edge, float(K[0,0]), float(K[1,1]) )
                    _TP, _FP, _Total = EvaluateAPof2D(depth, outline, marker, gt_marker, min_iou)
                    TP += _TP
                    FP += _FP
                    Total += _Total
                pre = float(TP) / max(TP + FP, 1e-8)
                rec = float(TP) / max(Total, 1e-8)
                if name=='valid':
                    print('Epoch %d/%d : precision/recall = %f,%f : n_converge=%d' % (model.epoch+1, n_epoch, pre, rec, n_converge) )
                log_pick['min_iou'] = min_iou
                log_pick['%s_ap'%name].append(pre)
                log_pick['%s_ar'%name].append(rec)

            log_pick['iter'].append(model.niter)
            log_pick['ep'].append(model.epoch+1)
            with open('unet_train_log_%s.pick'%dataset_name,'wb') as f:
                pickle.dump(log_pick, f, protocol=2)
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

            if log_pick['valid_ap'][-1] > .9 and\
               log_pick['valid_ar'][-1] > .9 and\
               log_pick['train_ap'][-1] > .9 and\
               log_pick['train_ar'][-1] > .9:
               n_converge += 1
            else:
               n_converge = 0

            if n_converge >= 4:
                break_condition = True
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type',
                    default='t',
                    choices=['t', 'e'])

    parser.add_argument('dataset_name', type=str, help='The name of output dataset')

    args = parser.parse_args()
    if args.type == 't':
        #Train(dataset_name='vtk_dataset_separated')
        Train(args.dataset_name)
    else:
        #Evaluate(dataset_name='vtk_dataset_separated')
        Evaluate(args.dataset_name)
