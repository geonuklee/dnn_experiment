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
            for each in eval_frame:
                eval_scene.append( (base,nframe)+ each)
            #a = GetColoredLabel(pred_marker)
            #b = GetColoredLabel(pick['marker'])
            #pred_gt = cv2.addWeighted(a,.5,b,.5,0.)
            #cv2.imshow("pred_gt", pred_gt)
            cv2.imshow("frame", dst)
            if ord('q') == cv2.waitKey(1):
                exit(1)
            fn = osp.join(screenshot_dir, 'frame_%d_%s_%04d.png'%(i_file,base,nframe) )
            cv2.imwrite(fn, dst)
            nframe += 1
            depth_msg, rgb_msg = None, None
            if nframe >= 20:
                break

        # TODO 통계 처리
        dtype = [('base',object), ('fidx',int),
                ('gidx',int),
                ('iou',float), ('recall',float), ('overseg',bool),('underseg',bool),
                ('pidx',int), ('precision',float)]
        eval_scene = np.array(eval_scene, dtype)
        dst = visualize_scene(pick,eval_scene)
        cv2.imshow("scene%d"%i_file, dst)
        if ord('q') == cv2.waitKey(1):
            exit(1)
        fn = osp.join(screenshot_dir, 'scene_%d_%s.png'%(i_file,base) )
        cv2.imwrite(fn, dst)


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
