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

from evaluator import Evaluator, SceneEval, FrameEval # TODO change file
from ros_client import *
from unet.gen_obblabeling import GetInitFloorMask

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

if __name__=="__main__":
    pkg_dir = '/home/geo/catkin_ws/src/ros_unet' # TODO Hard coding for now
    #obbdatasetpath = osp.join(pkg_dir,'obb_dataset_train','*.pick')
    obbdatasetpath = osp.join(pkg_dir,'obb_dataset_test','*.pick')
    gt_files = glob2.glob(obbdatasetpath)

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

    #cameras = ['cam0', 'cam1'] # For multiple camera test.

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

            #depth_msg = None
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
                #init_floormask  = get_init_floormask(bridge,
                #        rect_depth_msg.width, rect_depth_msg.height, y0=50)
                #floor_msg = compute_floor(rect_depth_msg, rect_rgb_msg, init_floormask)
                #plane_c = floor_msg.plane
                Twc = get_Twc(cam_id)
                scene_eval = SceneEval(pick, Twc, plane_c, max_z, cam_id)
                scene_eval.floor = floor
                evaluator.PutScene(pick['fullfn'],scene_eval)
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

            print("scene %d/%d ... %s "% (i_file, len(gt_files), gt_fn) )
            if n%10==0 :  # TODO
                evaluator.Evaluate(is_final=False)
                break

        # Draw for after evaluating a rosbag file.
        if evaluator.n_frame > 0:
            evaluator.Evaluate(is_final=False)
            print('Evaluate files.. %d/%d'%(i_file, len(gt_files)) )

    if evaluator.n_frame > 0:
        print("~~~~~~~~~~Final evaluation~~~~~~~~~~~")
        evaluator.Evaluate(is_final=True)

