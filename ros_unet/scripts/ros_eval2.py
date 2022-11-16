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

from evaluator import get_pkg_dir, get_pick
from ros_client import *
from unet.gen_obblabeling import GetInitFloorMask
from os import makedirs
import matplotlib.pyplot as plt
import pyautogui
import shutil
from unet.util import GetColoredLabel
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

def Evaluate2D(obb_resp, gt_marker, rgb):
    pred_marker = np.frombuffer(obb_resp.marker.data, dtype=np.int32)\
            .reshape(obb_resp.marker.height, obb_resp.marker.width)
    pred_filtered_outline = np.frombuffer(obb_resp.filtered_outline.data, dtype=np.uint8)\
            .reshape(obb_resp.marker.height, obb_resp.marker.width)
    # ref : https://stackoverflow.com/questions/24780697/numpy-unique-list-of-colors-in-the-image
    gt_pred = np.stack((gt_marker, pred_marker), axis=2)
    pair_marker, counts = np.unique(gt_pred.reshape(-1,2),axis=0, return_counts=True)
    arr = []
    for (gidx, pidx), count in zip(pair_marker,counts):
        if 0 in [gidx, pidx]:
            continue
        arr.append( (gidx, pidx, count) )
    arr = np.array(arr, dtype=[('gidx',int),('pidx',int),('count',float)] )

    undersegs, oversegs = set(), set()
    ious, recalls, max_precisions, g2p = {},{},{},{}

    gt_indices, tmp = np.unique(gt_marker, return_counts=True)
    gt_areas ={}
    min_th = .1
    for (gidx, gs) in zip(gt_indices, tmp.astype(np.float) ):
        gt_areas[gidx] = gs
        if gidx<1:
            continue
        matches = arr[arr['gidx']==gidx]
        matches = np.flip( matches[np.argsort(matches, order='count')] )
        if matches.shape[0] == 0:
            recall = 0.
            iou = 0.
            pidx = -1
        else:
            recall = matches['count'][0]/gs
            overlap_pred = []
            for pidx, s in matches[['pidx','count']]:
                if pidx < 1:
                    continue
                prec = s / gs
                if prec > min_th:
                    overlap_pred.append(pidx)
            if len(overlap_pred) > 1:
                oversegs.add(gidx)
            gt_pred_s = np.logical_or(gt_marker==gidx,pred_marker==pidx).sum()
            iou = matches['count'][0]/gt_pred_s
            pidx = matches['pidx']
        ious[gidx], recalls[gidx], max_precisions[gidx], g2p[gidx] = iou, recall, 0., pidx

    pred_indices, pred_areas = np.unique(pred_marker, return_counts=True)
    for (pidx, ps) in zip(pred_indices, pred_areas.astype(np.float) ):
        if pidx == 0:
            continue
        matches = arr[arr['pidx']==pidx]
        matches = np.flip( matches[np.argsort(matches, order='count')] )
        if matches.shape[0] == 0:
            prec = 0.
        else:
            prec = matches['count'][0] / ps
            overlap_gt = []
            for gidx, s in matches[['gidx','count']]:
                if gidx < 1:
                    continue
                recall = s/gt_areas[gidx]
                if recall > min_th:
                    overlap_gt.append(gidx)
            if len(overlap_gt) > 1:
                for gidx in overlap_gt:
                    undersegs.add(gidx)
        max_precisions[gidx] = max(max_precisions[gidx], prec)

    # TODO Visualization of evaluation for 'debugging'
    outputlist = []
    dst = np.zeros((gt_marker.shape[0],gt_marker.shape[1],3), dtype=np.uint8)
    #dst[gt_marker<1,:] = 0
    for gidx in gt_indices:
        if gidx == 0:
            continue
        recall, iou, pidx, precision = recalls[gidx], ious[gidx], g2p[gidx], max_precisions[gidx]
        overseg, underseg = gidx in oversegs, gidx in undersegs
        output = gidx, recall, iou, overseg, underseg
        outputlist.append(output)
        part = gt_marker==gidx
        if overseg:
            dst[part,0] = 255
        if underseg:
            dst[part,-1] = 255
        if not overseg and not underseg:
            dst[part,:] = rgb[part,:]
    boundary = GetBoundary(gt_marker, 2)
    dst_rgb = rgb.copy()
    dst_rgb[boundary>0,:] = dst[boundary>0,:] = 0
    dst = cv2.addWeighted(dst_rgb, .3, dst, .7, 0.)
    #cv2.imshow("gt",   GetColoredLabel(gt_marker,True))
    cv2.imshow("pred", GetColoredLabel(pred_marker))
    cv2.imshow("dst", dst)
    if ord('q') == cv2.waitKey(1):
        exit(1)
    return output

def perform_test(eval_dir, gt_files, screenshot_dir):
    if osp.exists(screenshot_dir):
        shutil.rmtree(screenshot_dir)
    makedirs(screenshot_dir)
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

    for i_file, gt_fn in enumerate(gt_files):
        print(gt_fn)
        pick = get_pick(gt_fn)
        bag, set_depth, set_rgb, topic2cam, rgb_topics, depth_topics, rgb_msgs, depth_msgs,\
                rect_info_msgs, mx, my, fx, fy, Twc, plane_c, floor = \
                get_topics(bridge,pkg_dir,gt_fn, pick, set_camera,floordetector_set_camera, compute_floor)

        secene_evallist, nframe = [], 0
        for topic, msg, t in bag.read_messages(topics=rgb_topics.values()+depth_topics.values()):
            cam_id = topic2cam[topic]
            if topic in set_depth:
                depth_msgs[cam_id] = msg
            elif topic in set_rgb:
                rgb_msgs[cam_id] = msg
                continue # 매 depth topic에 대해서만 OBB 생성후 evaluation 실행
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
            output = Evaluate2D(obb_resp, pick['marker'], rect_rgb)
            secene_evallist += output
            nframe += 1
            depth_msg, rgb_msg = None, None
            if nframe >= 5:
                break
        # TODO 통계 처리





def test_evaluation():
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_test0523')
    if not osp.exists(eval_dir):
        makedirs(eval_dir)
    profile_fn = osp.join(eval_dir, 'profile.pick')
    usages = ['test0523']
    gt_files = []
    for usage in usages:
        obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
        gt_files += glob2.glob(obbdatasetpath)

    perform_test(eval_dir, gt_files, osp.join(eval_dir, 'screenshot'))

if __name__=="__main__":
    test_evaluation()
    print("#######Evaluation is finished########")
