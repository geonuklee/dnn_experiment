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

from evaluator import *
from ros_client import *
from unet.gen_obblabeling import GetInitFloorMask
from os import makedirs
import matplotlib.pyplot as plt
import pyautogui
import shutil

def get_pick(fn):
    f = open(fn,'r')
    pick = pickle.load(f)
    f.close()
    return pick

def get_topicnames(bagfn, bag, given_camid='cam0'):
    depth = '/%s/helios2/depth/image_raw'%given_camid
    info  = '/%s/helios2/camera_info'%given_camid
    rgb   = '/%s/aligned/rgb_to_depth/image_raw'%given_camid
    return rgb, depth, info

def get_camid(fn):
    base = osp.splitext( osp.basename(fn) )[0]
    groups = re.findall("(.*)_(cam0|cam1)", base)[0]
    return groups[1]

def perform_test(gt_files, screenshot_dir):
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
    evaluator = Evaluator()
    for i_file, gt_fn in enumerate(gt_files):
        pick = get_pick(gt_fn)
        bag = rosbag.Bag(pick['fullfn'])
        rgb_topics, depth_topics, info_topics = {},{},{}
        rect_info_msgs = {}
        remap_maps = {}
        cameras = [get_camid(gt_fn)] # For each file test.
        for cam_id in cameras:
            rgb_topics[cam_id], depth_topics[cam_id], info_topics[cam_id] \
                    = get_topicnames(pick['fullfn'], bag, given_camid=cam_id)
            try:
                _, rgb_msg, _ = bag.read_messages(topics=[rgb_topics[cam_id]]).next()
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

        scene_eval = None
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
            fx, fy = rect_info_msgs[cam_id].K[0], rect_info_msgs[cam_id].K[4]
            rect_rgb_msg, rect_depth_msg, rect_depth = rectify(rgb_msg, depth_msg, mx, my, bridge)

            if scene_eval is None:
                cv_gt = cv2.imread(pick['cvgt_fn'])
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
                scene_eval = SceneEval(pick, Twc, plane_c, max_z, cam_id)
                scene_eval.floor = floor
                evaluator.PutScene(pick['fullfn'],scene_eval)
                scene_eval.pubGtObb()
                rate.sleep()

            rect_depth[floor>0] = 0.
            rect_depth_msg = bridge.cv2_to_imgmsg(rect_depth,encoding='32FC1')

            t0 = time.time()
            edge_resp = predict_edge(rect_rgb_msg,rect_depth_msg, fx, fy)
            plane_w = convert_plane(Twc, plane_c) # empty plane = no floor filter.
            obb_resp = compute_obb(rect_depth_msg, rect_rgb_msg, edge_resp.mask,
                    Twc, std_msgs.msg.String(cam_id), fx, fy, plane_w)
            t1 = time.time()
            frame_eval = FrameEval(scene_eval, cam_id, t1-t0, verbose=True)
            frame_eval.GetMatches(obb_resp.output)
            n = evaluator.PutFrame(pick['fullfn'], frame_eval)
            rate.sleep()

            base_bag = osp.splitext(osp.basename(pick['fullfn']))[0]
            im_screenshot = pyautogui.screenshot()
            im_screenshot = cv2.cvtColor(np.array(im_screenshot), cv2.COLOR_RGB2BGR)
            im_screenshot = cv2.resize( im_screenshot,(1200,800) )
            fn_screenshot = osp.join(screenshot_dir, '%04d_%s.png'%(evaluator.n_frame, base_bag) )
            cv2.imwrite(fn_screenshot, im_screenshot)

            print("scene %d/%d ... %s "% (n, len(gt_files), gt_fn) )
            evaluator.Evaluate(is_final=False)
            if n%5==0 :  # TODO
                break
        # Draw for after evaluating a rosbag file.
        if evaluator.n_frame > 0:
            evaluator.Evaluate(is_final=False)
            print('Evaluate files.. %d/%d'%(i_file, len(gt_files)) )
    plt.close()
    return evaluator

def get_pkg_dir():
    return osp.abspath( osp.join(osp.dirname(__file__),'..') )

def yaw_evaluation():
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_yaw')
    if not osp.exists(eval_dir):
         makedirs(eval_dir)
    profile_fn = osp.join(eval_dir, 'profile.pick')
    if not osp.exists(profile_fn):
        usages = ['alignedyaw']
        gt_files = []
        for usage in usages:
            obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
            gt_files += glob2.glob(obbdatasetpath)
        evaluator = perform_test(gt_files, osp.join(eval_dir, 'screenshot'))
        arr_scense, arr_frames, arr = evaluator.GetTables()
        with open(profile_fn,'wb') as f:
            pickle.dump({ 'arr_scense':arr_scense, 'arr_frames':arr_frames, 'arr':arr, }, f)
    else:
        with open(profile_fn,'rb') as f:
            pick = pickle.load(f)
            arr_scense, arr_frames, arr = pick['arr_scense'], pick['arr_frames'], pick['arr']
        evaluator = Evaluator()
    # TODO Draw yaw - histogram only
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1, 1)
    ax.title.set_text('skew angle')
    num_bins = 4
    tp = np.logical_and(arr['maxIoU'] > tp_iou, arr['crosscheck'])
    ## TODO under seg 감지가 안되나.
    DrawOverUnderHistogram(ax, num_bins, (0., 60.),
                tp_iou, tp, arr, arr['degskew_gt'], '[deg]')
    plt.savefig( osp.join(eval_dir, '%s.png'%osp.basename(eval_dir) ) )
    plt.close()

def dist_evaluation():
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_dist')
    if not osp.exists(eval_dir):
         makedirs(eval_dir)
    profile_fn = osp.join(eval_dir, 'profile.pick')
    if not osp.exists(profile_fn):
        usages = ['aligneddist']
        gt_files = []
        for usage in usages:
            obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
            gt_files += glob2.glob(obbdatasetpath)
        evaluator = perform_test(gt_files, osp.join(eval_dir, 'screenshot'))
        arr_scense, arr_frames, arr = evaluator.GetTables()
        with open(profile_fn,'wb') as f:
            pickle.dump({ 'arr_scense':arr_scense, 'arr_frames':arr_frames, 'arr':arr, }, f)
    else:
        with open(profile_fn,'rb') as f:
            pick = pickle.load(f)
            arr_scense, arr_frames, arr = pick['arr_scense'], pick['arr_frames'], pick['arr']
        evaluator = Evaluator()
    # TODO Draw yaw - histogram only
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1, 1)
    num_bins = 5
    tp = np.logical_and(arr['maxIoU'] > tp_iou, arr['crosscheck'])
    ax = fig.add_subplot(1,1, 1)
    ax.title.set_text('center depth')
    DrawOverUnderHistogram(ax, num_bins, (.8, arr['z_gt'].max()),
            tp_iou, tp, arr, arr['z_gt'], '[m]')
    plt.savefig(osp.join(eval_dir, '%s.png'%osp.basename(eval_dir) ) )
    plt.close()

def test_evaluation():
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_test0523')
    if not osp.exists(eval_dir):
         makedirs(eval_dir)
    profile_fn = osp.join(eval_dir, 'profile.pick')
    if not osp.exists(profile_fn):
        usages = ['test0523']
        gt_files = []
        for usage in usages:
            obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
            gt_files += glob2.glob(obbdatasetpath)
        evaluator = perform_test(gt_files, osp.join(eval_dir, 'screenshot'))
        arr_scense, arr_frames, arr = evaluator.GetTables()
        with open(profile_fn,'wb') as f:
            pickle.dump({ 'arr_scense':arr_scense, 'arr_frames':arr_frames, 'arr':arr, }, f)
    else:
        with open(profile_fn,'rb') as f:
            pick = pickle.load(f)
            arr_scense, arr_frames, arr = pick['arr_scense'], pick['arr_frames'], pick['arr']
        evaluator = Evaluator()
    evaluator.Evaluate(arr_scense,arr_frames,arr, is_final=False )
    plt.savefig(osp.join(eval_dir, '%s.png'%osp.basename(eval_dir) ) )

def roll_evaluation():
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_roll')
    if not osp.exists(eval_dir):
         makedirs(eval_dir)
    profile_fn = osp.join(eval_dir, 'profile.pick')
    if not osp.exists(profile_fn):
        usages = ['alignedroll']
        gt_files = []
        for usage in usages:
            obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
            gt_files += glob2.glob(obbdatasetpath)
        evaluator = perform_test(gt_files, osp.join(eval_dir, 'screenshot'))
        arr_scense, arr_frames, arr = evaluator.GetTables()
        with open(profile_fn,'wb') as f:
            pickle.dump({ 'arr_scense':arr_scense, 'arr_frames':arr_frames, 'arr':arr, }, f)
    else:
        with open(profile_fn,'rb') as f:
            pick = pickle.load(f)
            arr_scense, arr_frames, arr = pick['arr_scense'], pick['arr_frames'], pick['arr']
        evaluator = Evaluator()
    evaluator.Evaluate(arr_scense,arr_frames,arr, is_final=False )
    plt.savefig(osp.join(eval_dir, '%s.png'%osp.basename(eval_dir) ) )


if __name__=="__main__":
    yaw_evaluation()
    #dist_evaluation()
    #roll_evaluation()
    #test_evaluation()
