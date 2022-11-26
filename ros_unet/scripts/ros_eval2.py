#!/usr/bin/python2
#-*- coding:utf-8 -*-

import rospy
import pickle
import glob2 # For recursive glob for python2
import re
import os
from os import path as osp
import numpy as np
import rosbag
import ros_unet.srv

import sensor_msgs, std_msgs
import geometry_msgs
import cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as rotation_util
from collections import OrderedDict as Od

from evaluator import get_pkg_dir, get_pick
from ros_client import *
from unet.gen_obblabeling import GetInitFloorMask
from os import makedirs
import matplotlib.pyplot as plt
import pyautogui
import shutil
from unet.util import GetColoredLabel, Evaluate2D
from unet_ext import GetBoundary

def get_topicnames(bagfn, bag, given_camid='cam0'):
    depth = '/%s/helios2/depth/image_raw'%given_camid
    info  = '/%s/helios2/camera_info'%given_camid
    rgb   = '/%s/aligned/rgb_to_depth/image_raw'%given_camid
    return rgb, depth, info

def get_camid(fn):
    base = osp.splitext( osp.basename(fn) )[0]
    groups = re.findall("(.*)_(cam0|cam1)", base)[0]
    return groups[1]

def get_topics(bridge, pkg_dir,gt_fn, pick, set_camera,floordetector_set_camera, compute_floor):
    rosbag_fn = osp.join(pkg_dir, pick['rosbag_fn'] )
    bag = rosbag.Bag(rosbag_fn)
    rgb_topics, depth_topics, info_topics = {},{},{}
    rect_info_msgs = {}
    remap_maps = {}
    cameras = [get_camid(gt_fn)] # For each file test.
    for cam_id in cameras:
        rgb_topics[cam_id], depth_topics[cam_id], info_topics[cam_id] \
                = get_topicnames(rosbag_fn, bag, given_camid=cam_id)
        try:
            _, rgb_msg, _ = bag.read_messages(topics=[rgb_topics[cam_id]]).next()
            _, depth_msg, _ = bag.read_messages(topics=[depth_topics[cam_id]]).next()
            _, info_msg, _= bag.read_messages(topics=[info_topics[cam_id]]).next()
        except:
            continue
        rect_info_msgs[cam_id], mx, my = get_rectification(info_msg)
        remap_maps[cam_id] = (mx, my)
        set_camera(std_msgs.msg.String(cam_id), rect_info_msgs[cam_id])
        floordetector_set_camera(std_msgs.msg.String(cam_id), rect_info_msgs[cam_id])
    rgb_msgs, depth_msgs  = {}, {}
    topic2cam = {}
    for k,v in rgb_topics.items():
        rgb_msgs[k] = None
        topic2cam[v] = k
    for k,v in depth_topics.items():
        depth_msgs[k] = None
        topic2cam[v] = k
    set_depth = set(depth_topics.values())
    set_rgb = set(rgb_topics.values())
    fx, fy = rect_info_msgs[cam_id].K[0], rect_info_msgs[cam_id].K[4]

    rect_rgb_msg, rect_depth_msg, rect_depth, rect_rgb = rectify(rgb_msg, depth_msg, mx, my, bridge)
    cvgt_fn = osp.join(pkg_dir,pick['cvgt_fn'])
    cv_gt = cv2.imread(cvgt_fn)
    max_z = 5.
    init_floormask = GetInitFloorMask(cv_gt)
    if init_floormask is None:
        plane_c = (0., 0., 0., 99.)
        floor = np.zeros((rect_depth_msg.height,rect_depth_msg.width),np.uint8)
    else:
        floor_msg = compute_floor(rect_depth_msg, rect_rgb_msg, init_floormask)
        plane_c  = floor_msg.plane
        floor_mask = floor_msg.mask
        floor = np.frombuffer(floor_mask.data, dtype=np.uint8).reshape(floor_mask.height, floor_mask.width)
    Twc = get_Twc(cam_id)

    bag = rosbag.Bag(rosbag_fn)
    return bag, set_depth, set_rgb, topic2cam, rgb_topics, depth_topics, rgb_msgs, depth_msgs,\
            rect_info_msgs, mx, my, fx, fy, Twc, plane_c, floor

def visualize_scene(pick, eval_scene):
    gt_indices = np.unique(eval_scene['gidx'])
    dst = np.zeros_like(pick['rgb'])
    gt_marker = pick['marker']
    min_iou = .6
    msg = ' # /AP>%.1f/  Over / Under '%min_iou
    font_face, font_scale, font_thick = cv2.FONT_HERSHEY_PLAIN, 1., 1
    w,h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
    hoffset = 10
    w,h = w+2,h+hoffset
    dst_score = np.zeros((dst.shape[0],w,3),dst.dtype)
    cp = [0,h]
    cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, (255,255,255), font_thick)
    cp[1] += h+hoffset
    for gidx in gt_indices:
        data = eval_scene[eval_scene['gidx']==gidx]
        n = len(data)
        n_underseg = data['underseg'].sum()
        n_overseg = data['overseg'].sum()
        n_ap = (data['iou']> min_iou).sum()
        prob_underseg = float(n_underseg) / float(n)
        prob_overseg  = float(n_overseg) / float(n)
        ap = float(n_ap) / float(n)
        part = gt_marker == gidx
        if n_overseg > 0:
            dst[part,0] = 255
        if n_underseg > 0:
            dst[part,-1] = 255
        #print(gidx, n_overseg, n_underseg)
        if not prob_overseg and not prob_underseg:
            dst[part,:] = pick['rgb'][part,:]
        #msg = '%2d / %.3f / %.3f / %.3f' %(gidx, ap, prob_overseg, prob_underseg)
        cp[0] = 0
        msg = '%2d' % gidx
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, (255,255,255), font_thick)

        if ap < 1.:
            color = (0,0,255)
        else:
            color = (255,255,255)
        cp[0] += w
        msg = '   %.3f'%ap
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, color, font_thick)

        if n_overseg > 0:
            color = (0,0,255)
        else:
            color = (255,255,255)
        cp[0] += w
        msg = '   %.3f'%prob_overseg
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, color, font_thick)

        if n_underseg > 0:
            color = (0,0,255)
        else:
            color = (255,255,255)
        cp[0] += w
        msg = '   %.3f'%prob_underseg
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, color, font_thick)
        cp[1] += h+hoffset
        #cp = (cp[0], cp[1]+h)
        #cv2.putText(dst_score, msg, cp, font_face, font_scale, (255,255,255), font_thick)
        #h += 10

    boundary = GetBoundary(gt_marker, 2)
    dst_rgb = pick['rgb'].copy()
    dst_rgb[boundary>0,:] = dst[boundary>0,:] = 0
    dst = cv2.addWeighted(dst_rgb, .3, dst, .7, 0.)

    for gidx in gt_indices:
        part = gt_marker == gidx
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats( part.astype(np.uint8) )
        for i, (x0,y0,w,h,s) in enumerate(stats):
            if w == gt_marker.shape[1] and h == gt_marker.shape[0]:
                continue
            pt = centroids[i].astype(np.int)
            msg = '%d'%gidx
            #w,h = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN,1.5,2)
            w,h = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN,1.5,2)[0]
            cv2.rectangle(dst,(pt[0]-2,pt[1]+5),(pt[0]+w+2,pt[1]-h-5),(255,255,255),-1)
            cv2.rectangle(dst,(pt[0]-2,pt[1]+5),(pt[0]+w+2,pt[1]-h-5),(100,100,100),1)
            cv2.putText(dst, msg, (pt[0],pt[1]), cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,0),2)

    dst = np.hstack((dst,dst_score))
    return dst

def GetOblique(eval_data, picks):
    oblique_array = np.repeat(-1.,eval_data.shape)
    depthvec = np.array((0.,0.,1.))
    for base, pick in picks.items():
        obbs = {}
        for each in pick['obbs']:
            obbs[each['id']] = each
        gt_marker = pick['marker']
        w,h = float(gt_marker.shape[1]), float(gt_marker.shape[0])
        for gidx in np.unique(gt_marker):
            if gidx == 0:
                continue
            indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
            t_cb = obbs[gidx]['pose'][:3]
            qw,qx,qy,qz = obbs[gidx]['pose'][3:] # w,x,y,z
            rot_cb = rotation_util.from_quat([qx, qy, qz, qw])
            nvec = rot_cb.as_dcm()[:,0]
            deg = np.rad2deg(np.arccos(-nvec.dot(depthvec)))
            oblique_array[indicies] = deg
    assert( (oblique_array<0).sum() == 0)
    return oblique_array

def GetDistance(eval_data, picks):
    distance_array = np.repeat(-1.,eval_data.shape)
    for base, pick in picks.items():
        obbs = {}
        for each in pick['obbs']:
            obbs[each['id']] = each
        gt_marker = pick['marker']
        w,h = float(gt_marker.shape[1]), float(gt_marker.shape[0])
        for gidx in np.unique(gt_marker):
            if gidx == 0:
                continue
            indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
            t_cb = obbs[gidx]['pose'][:3]
            distance_array[indicies] = np.linalg.norm(t_cb)
            #distance_array[indicies] = t_cb[2]
    assert( (distance_array<0).sum() == 0)
    return distance_array 

def GetMargin(eval_data, picks):
    margin_array = np.repeat(-1,eval_data.shape)
    minwidth_array = np.repeat(-1,eval_data.shape)
    for base, pick in picks.items():
        gt_marker = pick['marker']
        w,h = float(gt_marker.shape[1]), float(gt_marker.shape[0])
        for gidx in np.unique(gt_marker):
            if gidx == 0:
                continue
            indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
            part = gt_marker==gidx
            part[0,:] = part[:,0] = part[-1,:] = part[:,-1] = 0
            dist_part = cv2.distanceTransform( part.astype(np.uint8),
                    distanceType=cv2.DIST_L2, maskSize=5)
            loc = np.unravel_index( np.argmax(dist_part,axis=None), gt_marker.shape)
            a = min(loc[0], h-loc[0])
            b = min(loc[1], w-loc[1])
            d = min( float(a), float(b) )
            margin_array[indicies] = d
            minwidth_array[indicies] = dist_part[loc]
    assert( (margin_array<0).sum() == 0)
    return margin_array, minwidth_array

def GetTags(eval_data, picks, tags):
    tag_array = np.repeat('',eval_data.shape).astype(object)
    for base, pick in picks.items():
        gt_marker = pick['marker']
        for gidx in np.unique(gt_marker):
            if gidx == 0:
                continue
            indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
            key = (base,gidx)
            if key in tags:
                tag_array[indicies] = tags[key]
    return tag_array

    #valid = np.repeat(True,eval_data.shape)
    #for base, pick in picks.items():
    #    gt_marker = pick['marker']
    #    for gidx in np.unique(gt_marker):
    #        if gidx == 0:
    #            continue
    #        indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
    #        if (base,gidx) in tags: # tag = tags[(base,gidx)]
    #            valid[indicies] = False
    #return valid

def PlotMarginAp(eval_data, picks, margin, valid, fig, ax, min_iou,
        num_bins=5, min_max=(0., 100.),
        show_underseg=False, show_overseg=False):
    la = np.logical_and
    n_hist , bound    = np.histogram(margin[valid], num_bins,min_max)
    tp_hist , _       = np.histogram(margin[la(eval_data['iou']>min_iou,valid)], num_bins,min_max)
    underseg_hist , _ = np.histogram(margin[la(eval_data['underseg'],valid)], num_bins,min_max)
    overseg_hist , _  = np.histogram(margin[la(eval_data['overseg'],valid)], num_bins,min_max)
    no_samples = n_hist==0
    n_hist[no_samples] = 1 # To prevent divide by zero
    nbar= 1
    if show_underseg:
        if underseg_hist.sum()==0:
            show_underseg = False
        else:
            nbar+=1
    if show_overseg:
        if overseg_hist.sum()==0:
            show_overseg = False
        else:
            nbar+=1
    tp_hist = tp_hist.astype(float) / n_hist.astype(float)
    underseg_hist = underseg_hist.astype(float) / n_hist.astype(float)
    overseg_hist  = overseg_hist.astype(float) / n_hist.astype(float)
    n_hist[no_samples] = 0
    width = 1. / float(nbar) - .1
    x = np.arange(num_bins)
    offset = float(nbar-1)*width/2.
    ap_label = 'AP (IoU >%.1f)'%min_iou
    ax.bar(x-offset, width=width, height=tp_hist, alpha=.5, label=ap_label)
    offset -= width
    if show_underseg:
        ax.bar(x-offset, width=width, height=underseg_hist, alpha=.5, label='Under-segment')
        offset -= width
    if show_overseg:
        ax.bar(x-offset, width=width, height=overseg_hist, alpha=.5, label='Over-segment')
        offset -= width
    xlabels = []
    for i in range(num_bins):
        msg = '%.1f~%.1f'%(bound[i],bound[i+1])
        msg += '\nn(instace)=%d'%n_hist[i]
        xlabels.append(msg)
    ax.set_xlabel('Margin [pixel]',rotation=0, fontsize=7, fontweight='bold')
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=7)
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.set_xticks(x)
    ax.yaxis.set_label_coords(-0.08, 1.)
    if nbar > 1:
        ax.legend(loc='upper left', fontsize=7)
    else:
        ax.set_ylabel(ap_label,rotation=0, fontsize=7, fontweight='bold')
    return

def PlotMinwidthAp(eval_data, picks, minwidth, valid, fig, ax, min_iou,
        num_bins=9, min_max=(10., 100.),
        show_underseg=False, show_overseg=False):
    la = np.logical_and
    n_hist , bound    = np.histogram(minwidth[valid], num_bins,min_max)
    tp_hist , _       = np.histogram(minwidth[la(eval_data['iou']>min_iou,valid)], num_bins,min_max)
    underseg_hist , _ = np.histogram(minwidth[la(eval_data['underseg'],valid)], num_bins,min_max)
    overseg_hist , _  = np.histogram(minwidth[la(eval_data['overseg'],valid)], num_bins,min_max)
    no_samples = n_hist==0
    n_hist[no_samples] = 1 # To prevent divide by zero
    nbar= 1
    if show_underseg:
        if underseg_hist.sum()==0:
            show_underseg = False
        else:
            nbar+=1
    if show_overseg:
        if overseg_hist.sum()==0:
            show_overseg = False
        else:
            nbar+=1
    tp_hist = tp_hist.astype(float) / n_hist.astype(float)
    underseg_hist = underseg_hist.astype(float) / n_hist.astype(float)
    overseg_hist  = overseg_hist.astype(float) / n_hist.astype(float)
    n_hist[no_samples] = 0
    width = 1. / float(nbar) - .1
    x = np.arange(num_bins)
    offset = float(nbar-1)*width/2.
    ap_label = 'AP (IoU >%.1f)'%min_iou
    ax.bar(x-offset, width=width, height=tp_hist, alpha=.5, label=ap_label)
    offset -= width
    if show_underseg:
        ax.bar(x-offset, width=width, height=underseg_hist, alpha=.5, label='Under-segment')
        offset -= width
    if show_overseg:
        ax.bar(x-offset, width=width, height=overseg_hist, alpha=.5, label='Over-segment')
        offset -= width
    xlabels = []
    for i in range(num_bins):
        msg = '%.1f~%.1f'%(bound[i],bound[i+1])
        msg += '\nn(instace)=%d'%n_hist[i]
        xlabels.append(msg)
    ax.set_xlabel('Min width [pixel]',rotation=0, fontsize=7, fontweight='bold')
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=7)
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.set_xticks(x)
    ax.yaxis.set_label_coords(-0.08, 1.)
    if nbar > 1:
        ax.legend(loc='upper left', fontsize=7)
    else:
        ax.set_ylabel(ap_label,rotation=0, fontsize=7, fontweight='bold')
    return

def PlotIncircleRadius(eval_data, picks, margin, valid, fig, ax, min_iou):
    la = np.logical_and
    num_bins = 5
    min_max = (0., 100.)
    n_hist , bound  = np.histogram(margin[valid], num_bins,min_max)
    tp_hist , bound = np.histogram(margin[la(eval_data['iou']>min_iou,valid)], num_bins,min_max)
    no_samples = n_hist==0
    n_hist[no_samples] = 1 # To prevent divide by zero
    tp_hist = tp_hist.astype(float) / n_hist.astype(float)
    n_hist[no_samples] = 0
    x = np.arange(num_bins)

    ax.bar(x, width=.9, height=tp_hist, alpha=.5, label='TP segment')
    xlabels = []
    for i in range(num_bins):
        msg = '%.1f~%.1f'%(bound[i],bound[i+1])
        msg += '\nn(instace)=%d'%n_hist[i]
        xlabels.append(msg)
    ax.set_xlabel('Margin [pixel]',rotation=0, fontsize=7, fontweight='bold')
    ax.set_ylabel('AP(IoU > %.1f)'%min_iou,rotation=0, fontsize=7, fontweight='bold')
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=7)
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.set_xticks(x)
    ax.yaxis.set_label_coords(-0.08, 1.)
    #ax.legend(loc='lower right', fontsize=7)
    return 


def PlotObliqueAp(eval_data, picks, oblique, valid, fig, ax, min_iou,
        num_bins=5, min_max=(0., 50.),
        show_underseg=False, show_overseg=False):
    la = np.logical_and
    n_hist , bound  = np.histogram(oblique[valid], num_bins,min_max)
    tp_hist , _       = np.histogram(oblique[la(eval_data['iou']>min_iou,valid)], num_bins,min_max)
    underseg_hist , _ = np.histogram(oblique[la(eval_data['underseg'],valid)], num_bins,min_max)
    overseg_hist , _  = np.histogram(oblique[la(eval_data['overseg'],valid)], num_bins,min_max)
    no_samples = n_hist==0
    n_hist[no_samples] = 1 # To prevent divide by zero
    nbar= 1
    if show_underseg:
        if underseg_hist.sum()==0:
            show_underseg = False
        else:
            nbar+=1
    if show_overseg:
        if overseg_hist.sum()==0:
            show_overseg = False
        else:
            nbar+=1
    tp_hist = tp_hist.astype(float) / n_hist.astype(float)
    underseg_hist = underseg_hist.astype(float) / n_hist.astype(float)
    overseg_hist  = overseg_hist.astype(float) / n_hist.astype(float)
    n_hist[no_samples] = 0
    width = 1. / float(nbar) - .1
    x = np.arange(num_bins)
    offset = float(nbar-1)*width/2.
    ap_label = 'AP (IoU >%.1f)'%min_iou
    ax.bar(x-offset, width=width, height=tp_hist, alpha=.5, label=ap_label)
    offset -= width
    if show_underseg:
        ax.bar(x-offset, width=width, height=underseg_hist, alpha=.5, label='Under-segment')
        offset -= width
    if show_overseg:
        ax.bar(x-offset, width=width, height=overseg_hist, alpha=.5, label='Over-segment')
        offset -= width
    xlabels = []
    for i in range(num_bins):
        msg = '%.1f~%.1f'%(bound[i],bound[i+1])
        msg += '\nn(instace)=%d'%n_hist[i]
        xlabels.append(msg)
    ax.set_xlabel('Oblique [deg]',rotation=0, fontsize=7, fontweight='bold')
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=7)
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.set_xticks(x)
    ax.yaxis.set_label_coords(-0.08, 1.)
    if nbar > 1:
        ax.legend(loc='upper right', fontsize=7)
    else:
        ax.set_ylabel(ap_label,rotation=0, fontsize=7, fontweight='bold')
    return 

def PlotTagAp(eval_data, tags, fig, min_iou, show_underseg=False, show_overseg=False):
    la = np.logical_and
    def f_have(arr, word):
        return word in arr.split(';')
    have = np.vectorize(f_have)
    #valid = have(tags,'tape')
    nbar= 1
    if show_underseg:
        nbar+=1
    if show_overseg:
        nbar+=1
    width = 1. / float(nbar) - .1
    offset = float(nbar-1)*width/2.
    ax = fig.add_subplot(111)

    params = Od()
    params['AP (IoU >%.1f)'%min_iou] = eval_data['iou']>min_iou
    if show_underseg:
        params['underseg'] = eval_data['underseg']
    if show_overseg:
        params['overseg'] = eval_data['overseg']
    
    for name, inliers in params.items():
        case_nhist = Od()
        valid = tags==''
        nsample = valid.sum()
        case_nhist['No specifics\nn(instance)=%d'%nsample]\
                = float( la(inliers, valid).sum() ) / float(nsample)

        valid = have(tags,'tape')
        nsample = valid.sum()
        case_nhist['With tape\nn(instance)=%d'%nsample]\
                = float( la(inliers, valid).sum() ) / float(nsample)
        x = np.array(range(len(case_nhist))).astype(float) - offset
        rects = ax.bar(x,width=width,height=case_nhist.values(), alpha=.5, label=name)
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+.5*rect.get_width(), height,
                '%.2f'%height, ha='center', va='bottom')
        offset -= width

    ax.set_xticklabels(case_nhist.keys(), rotation=0.,fontsize=7)
    ax.set_xticks(range(len(case_nhist)))
    plt.tight_layout(rect=(.1, 0.,.95,.95)) # left,bottom,right,top
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    ax.legend(loc='best', fontsize=7)

    #ax.set_ylabel('[-]'%min_iou,rotation=0, fontweight='bold')
    ax.set_ylim(0.,1.)
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.yaxis.set_label_coords(-0.08, 1.05)
    return

def PlotDistanceAp(eval_data, picks, distance, valid, fig, ax, min_iou,
        num_bins=4, min_max=(.5, 3.),
        show_underseg=False, show_overseg=False):
    la = np.logical_and
    n_hist , bound    = np.histogram(distance[valid], num_bins,min_max)
    tp_hist , _       = np.histogram(distance[la(eval_data['iou']>min_iou,valid)], num_bins,min_max)
    underseg_hist , _ = np.histogram(distance[la(eval_data['underseg'],valid)], num_bins,min_max)
    overseg_hist , _  = np.histogram(distance[la(eval_data['overseg'],valid)], num_bins,min_max)
    no_samples = n_hist==0
    n_hist[no_samples] = 1 # To prevent divide by zero
    nbar= 1
    if show_underseg:
        if underseg_hist.sum()==0:
            show_underseg = False
        else:
            nbar+=1
    if show_overseg:
        if overseg_hist.sum()==0:
            show_overseg = False
        else:
            nbar+=1
    tp_hist = tp_hist.astype(float) / n_hist.astype(float)
    underseg_hist = underseg_hist.astype(float) / n_hist.astype(float)
    overseg_hist  = overseg_hist.astype(float) / n_hist.astype(float)
    n_hist[no_samples] = 0
    width = 1. / float(nbar) - .1
    x = np.arange(num_bins)
    offset = float(nbar-1)*width/2.
    ap_label = 'AP (IoU >%.1f)'%min_iou
    ax.bar(x-offset, width=width, height=tp_hist, alpha=.5, label=ap_label)
    offset -= width
    if show_underseg:
        ax.bar(x-offset, width=width, height=underseg_hist, alpha=.5, label='Under-segment')
        offset -= width
    if show_overseg:
        ax.bar(x-offset, width=width, height=overseg_hist, alpha=.5, label='Over-segment')
        offset -= width

    xlabels = []
    for i in range(num_bins):
        msg = '%.1f~%.1f'%(bound[i],bound[i+1])
        msg += '\nn(instace)=%d'%n_hist[i]
        xlabels.append(msg)
    ax.set_xlabel('Distance [m]',rotation=0, fontsize=7, fontweight='bold')
    ax.set_ylabel('AP(IoU > %.1f)'%min_iou,rotation=0, fontsize=7, fontweight='bold')
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=7)
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.set_xticks(x)
    ax.yaxis.set_label_coords(-0.08, 1.)
    if nbar > 1:
        ax.legend(loc='upper right', fontsize=7)
    else:
        ax.set_ylabel(ap_label,rotation=0, fontsize=7, fontweight='bold')
    return

def perform_test(eval_dir, gt_files,fn_evaldata):
    #if osp.exists(screenshot_dir):
    #    shutil.rmtree(screenshot_dir)
    #makedirs(screenshot_dir)
    rospy.init_node('evaluator', anonymous=True)
    rospy.wait_for_service('~PredictEdge')
    predict_edge = rospy.ServiceProxy('~PredictEdge', ros_unet.srv.ComputeEdge)
    rospy.wait_for_service('~SetCamera')
    set_camera = rospy.ServiceProxy('~SetCamera', ros_unet.srv.SetCamera)
    rospy.wait_for_service('~ComputeObb')
    compute_obb = rospy.ServiceProxy('~ComputeObb', ros_unet.srv.ComputeObb)
    bridge = CvBridge()
    rospy.wait_for_service('~FloorDetector/SetCamera')
    floordetector_set_camera = rospy.ServiceProxy('~FloorDetector/SetCamera', ros_unet.srv.SetCamera)
    rospy.wait_for_service('~FloorDetector/ComputeFloor')
    compute_floor = rospy.ServiceProxy('~FloorDetector/ComputeFloor', ros_unet.srv.ComputeFloor)
    rate = rospy.Rate(hz=50)
    pkg_dir = get_pkg_dir()

    eval_data = None
    for i_file, gt_fn in enumerate(gt_files):
        #print(gt_fn)
        pick = get_pick(gt_fn)
        base = osp.splitext(osp.basename(pick['rosbag_fn']))[0]
        bag, set_depth, set_rgb, topic2cam, rgb_topics, depth_topics, rgb_msgs, depth_msgs,\
                rect_info_msgs, mx, my, fx, fy, Twc, plane_c, floor = \
                get_topics(bridge,pkg_dir,gt_fn, pick, set_camera,floordetector_set_camera, compute_floor)

        eval_scene, nframe = [], 0
        for topic, msg, t in bag.read_messages(topics=rgb_topics.values()+depth_topics.values()):
            cam_id = topic2cam[topic]
            if topic in set_depth:
                depth_msgs[cam_id] = msg
            elif topic in set_rgb:
                rgb_msgs[cam_id] = msg

            rgb_msg, depth_msg = rgb_msgs[cam_id], depth_msgs[cam_id]
            if depth_msg is None or rgb_msg is None:
                continue
            rect_rgb_msg, rect_depth_msg, rect_depth, rect_rgb = rectify(rgb_msg, depth_msg, mx, my, bridge)
            #rect_depth[floor>0] = 0.
            #rect_depth_msg = bridge.cv2_to_imgmsg(rect_depth,encoding='32FC1')

            t0 = time.time()
            edge_resp = predict_edge(rect_rgb_msg,rect_depth_msg, fx, fy)
            plane_w = convert_plane(Twc, plane_c) # empty plane = no floor filter.
            obb_resp = compute_obb(rect_depth_msg, rect_rgb_msg, edge_resp.edge,
                    Twc, std_msgs.msg.String(cam_id), fx, fy, plane_w)
            t1 = time.time()

            #obb_resp.filtered_outline, obb_resp.marker, obb_resp.output
            eval_frame, pred_marker, dst = Evaluate2D(obb_resp, pick['marker'], rect_rgb)
            # TODO Evaluate3D obb_respoutput.markers <-> {obbs[k]['id']:obbs[k]['pose','scale']}
            for each in eval_frame:
                eval_scene.append( (base,i_file,nframe)+ each)
            cv2.imshow("frame", dst)
            if ord('q') == cv2.waitKey(1):
                exit(1)
            fn = osp.join(eval_dir, 'frame_%d_%s_%04d.png'%(i_file,base,nframe) )
            cv2.imwrite(fn, dst)
            nframe += 1
            depth_msg, rgb_msg = None, None
            if nframe >= 20:
                break

        dtype = [('base',object),
                ('sidx',int), # Scene index
                ('fidx',int), # Frame index
                ('gidx',int), # Ground truth object index
                ('iou',float), ('recall',float), ('overseg',bool),('underseg',bool),
                ('pidx',int), # Prediction object index
                ('precision',float)
                ]
        eval_scene = np.array(eval_scene, dtype)
        if eval_data is None:
            eval_data = eval_scene
        else:
            eval_data = np.hstack((eval_data,eval_scene) )
        dst = visualize_scene(pick,eval_scene)
        cv2.imshow("scene%d"%i_file, dst)
        if ord('q') == cv2.waitKey(1):
            exit(1)
        fn = osp.join(eval_dir, 'scene_%d_%s.png'%(i_file,base) )
        cv2.imwrite(fn, dst)

    cv2.destroyAllWindows()
    with open(fn_evaldata,'wb') as f: 
        np.save(f, eval_data)

    return eval_data

def test_evaluation():
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_test0523')
    if not osp.exists(eval_dir):
        makedirs(eval_dir)
    fn_evaldata = osp.join(eval_dir,'eval_data.npy')
    usages = ['test0523']
    gt_files = []
    for usage in usages:
        obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
        gt_files += glob2.glob(obbdatasetpath)
    if not osp.exists(fn_evaldata):
        eval_data = perform_test(eval_dir, gt_files, fn_evaldata)
    else:
        with open(fn_evaldata,'rb') as f: 
            eval_data = np.load(f, allow_pickle=True)
    picks = {}
    for fn in gt_files:
        base = osp.splitext( osp.basename(fn) )[0]
        groups = re.findall("(.*)_(cam0|cam1)", base)[0]
        base = groups[0]
        with open(fn,'r') as f:
            picks[base] = pickle.load(f)
    tags = {}
    tags['eval_test0523/scene_3_helios_test_2022-05-23-15-37-08.png']\
        = { 8:'tape'}
    tags['eval_test0523/scene_4_helios_test_2022-05-23-15-37-32.png']\
        = { 8:'tape'}
    tags['eval_test0523/scene_5_helios_2022-05-06-20-11-00.png'] = {11:'tape'}
    tags['eval_test0523/scene_6_helios_test_2022-05-23-15-52-03.png'] = {3:'tape'}
    tags['eval_test0523/scene_7_helios_scene2_2022-05-23-15-57-05.png']\
            = {6:'flat box', 9:'tape'}
    tmp_tags = {}
    for fn, val in tags.items():
        fn = osp.splitext( osp.basename(fn) )[0]
        base = re.findall("scene_\d*_(.*)", fn)[0]
        for gidx, tag in val.items():
            tmp_tags[(base,gidx)] = tag
    tags = tmp_tags
    
    la = np.logical_and
    oblique = GetOblique(eval_data, picks)
    margin, minwidth = GetMargin(eval_data, picks)
    distance = GetDistance(eval_data, picks)
    tags   = GetTags(eval_data, picks, tags)
    show = True

    N, axes, figs = 4, {}, {}
    for i in range(N):
        figs[i] = plt.figure(i+1, figsize=(12,3), dpi=100)
        axes[i] = figs[i].add_subplot(111)
        #fig.subplots_adjust(wspace=.1, hspace=.4)
    valid = tags==''
    PlotMarginAp(  eval_data, picks, margin,  valid, figs[0], axes[0], min_iou=.6, show_underseg=show)
    PlotMinwidthAp(eval_data, picks, minwidth,valid, figs[1], axes[1], min_iou=.6, show_underseg=show)
    PlotObliqueAp( eval_data, picks, oblique, valid, figs[2], axes[2], min_iou=.6, show_underseg=show)
    PlotDistanceAp(eval_data, picks, distance,valid, figs[3], axes[3], min_iou=.6, show_underseg=show)

    figs[0].savefig(osp.join(eval_dir,'test_margin_ap.svg'))
    figs[1].savefig(osp.join(eval_dir,'test_minwidth_ap.svg'))
    figs[2].savefig(osp.join(eval_dir,'test_oblique_ap.svg'))
    figs[3].savefig(osp.join(eval_dir,'test_distance_ap.svg'))

    fig = plt.figure(figsize=(4,3), dpi=100)
    PlotTagAp(eval_data, tags, fig, min_iou=.6, show_overseg=True, show_underseg=True)

    # TODO tape set과 normal set 사이의 AP, Over-seg 비교 
    plt.show(block=True)
    '''
    # TODO
        * [x] Margin <-> AP, prob(under seg)
        * [x] 기울기 <-> AP
        * [*] 깊이 거리 <-> AP, prob(under seg)
    * 표면상태와 인식률 
        * tags 추가후 범주별 AP, prob(over seg) 비교
    '''

    return

def dist_evaluation():
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_dist')
    if not osp.exists(eval_dir):
        makedirs(eval_dir)
    fn_evaldata = osp.join(eval_dir,'eval_data.npy')
    usages = ['aligneddist']
    gt_files = []
    for usage in usages:
        obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
        gt_files += glob2.glob(obbdatasetpath)
    if not osp.exists(fn_evaldata):
        eval_data = perform_test(eval_dir, gt_files, fn_evaldata)
    else:
        with open(fn_evaldata,'rb') as f: 
            eval_data = np.load(f, allow_pickle=True)
    picks = {}
    for fn in gt_files:
        base = osp.splitext( osp.basename(fn) )[0]
        groups = re.findall("(.*)_(cam0|cam1)", base)[0]
        base = groups[0]
        with open(fn,'r') as f:
            picks[base] = pickle.load(f)
    tags = {}
    distance = GetDistance(eval_data, picks)
    margin, minwidth = GetMargin(eval_data, picks)
    valid    = margin>40.
    fig = plt.figure(1, figsize=(12,4), dpi=100)
    ax = fig.add_subplot(111)
    PlotDistanceAp(eval_data, picks, distance, valid, fig, ax, min_iou=.5,
            num_bins=5,min_max=(1.,2.5), show_underseg=True, show_overseg=True)
    fig.savefig(osp.join(eval_dir,'dist_ap.svg') )
    plt.show(block=True)
    return

def oblique_evaluation():
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_oblique')
    if not osp.exists(eval_dir):
        makedirs(eval_dir)
    fn_evaldata = osp.join(eval_dir,'eval_data.npy')
    usages = ['alignedyaw', 'alignedroll']
    gt_files = []
    for usage in usages:
        obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
        gt_files += glob2.glob(obbdatasetpath)
    if not osp.exists(fn_evaldata):
        eval_data = perform_test(eval_dir, gt_files, fn_evaldata)
    else:
        with open(fn_evaldata,'rb') as f: 
            eval_data = np.load(f, allow_pickle=True)
    picks = {}
    for fn in gt_files:
        base = osp.splitext( osp.basename(fn) )[0]
        groups = re.findall("(.*)_(cam0|cam1)", base)[0]
        base = groups[0]
        with open(fn,'r') as f:
            picks[base] = pickle.load(f)
    tags = {}
    margin, minwidth = GetMargin(eval_data, picks)
    oblique = GetOblique(eval_data, picks)
    valid   = margin>40.
    fig = plt.figure(1, figsize=(12,4), dpi=100)
    ax = fig.add_subplot(111)
    PlotObliqueAp(eval_data,picks,oblique,valid,fig,ax,min_iou=.5, show_underseg=True)
    plt.show(block=True)
    fig.savefig(osp.join(eval_dir,'oblique_ap.svg'))
    return

if __name__=="__main__":
    test_evaluation()
    #dist_evaluation()
    #oblique_evaluation()

    print("#######Evaluation is finished########")
