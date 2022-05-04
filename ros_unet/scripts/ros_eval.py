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

from evaluator import Evaluator # TODO change file

def get_pick(fn):
    f = open(gt_fn,'r')
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

def get_rectification(camera_info):
    K = np.array( camera_info.K ,dtype=np.float).reshape((3,3))
    D = np.array( camera_info.D, dtype=np.float).reshape((-1,))

    osize = (camera_info.width,camera_info.height)
    newK,_ = cv2.getOptimalNewCameraMatrix(K,D,osize,1)
    mx,my = cv2.initUndistortRectifyMap(K,D,None,newK,osize,cv2.CV_32F)

    new_info = sensor_msgs.msg.CameraInfo()
    new_info.width = camera_info.width
    new_info.height = camera_info.height
    new_info.K = tuple(newK.reshape(-1,).tolist())
    new_info.D = (0,0,0,0,0)
    return new_info, mx, my

def convert2pose(Twc):
    pose = geometry_msgs.msg.Pose()

    Twc = np.array(Twc).reshape((4,4))
    Rwc = rotation_util.from_dcm(Twc[:3,:3])
    quat = Rwc.as_quat()
    twc = Twc[:3,3].reshape((3,))

    pose.position.x = twc[0]
    pose.position.y = twc[1]
    pose.position.z = twc[2]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose

def rectify(rgb_msg, depth_msg, mx, my, bridge):
    rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3)
    depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
    rgb = cv2.remap(rgb,mx,my,cv2.INTER_NEAREST)
    depth = cv2.remap(depth,mx,my,cv2.INTER_NEAREST)

    rect_rgb_msg = bridge.cv2_to_imgmsg(rgb,encoding=rgb_msg.encoding)
    rect_depth_msg = bridge.cv2_to_imgmsg(depth,encoding=depth_msg.encoding)
    return rect_rgb_msg, rect_depth_msg

def get_Twc(cam_id):
    Twc = rospy.get_param("/%s/base_T_cam"%cam_id)
    pose = convert2pose(Twc)
    return pose

if __name__=="__main__":
    pkg_dir = '/home/geo/catkin_ws/src/ros_unet' # TODO Hard coding for now
    obbdatasetpath = osp.join(pkg_dir,'obb_dataset','*.pick') # remove hardcoding .. 
    gt_files = glob2.glob(obbdatasetpath)

    rospy.init_node('evaluator', anonymous=True)
    rospy.wait_for_service('~PredictEdge')
    predict_edge = rospy.ServiceProxy('~PredictEdge', ros_unet.srv.ComputeEdge)

    rospy.wait_for_service('~SetCamera')
    set_camera = rospy.ServiceProxy('~SetCamera', ros_unet.srv.SetCamera)

    rospy.wait_for_service('~ComputeObb')
    compute_obb = rospy.ServiceProxy('~ComputeObb', ros_unet.srv.ComputeObb)
    bridge = CvBridge()
    
    #cameras = ['cam0', 'cam1'] # For multiple camera test.

    for gt_fn in gt_files:
        pick = get_pick(gt_fn)
        bag = rosbag.Bag(pick['fullfn'])
        rgb_topics, depth_topics, info_topics = {},{},{}
        rect_info_msgs = {}
        remap_maps = {}
        cameras = [get_camid(gt_fn)] # For each file test.
        evaluators = {}
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
            Twc = get_Twc(cam_id)
            evaluators[cam_id] = Evaluator(pick, Twc, cam_id)

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
            Twc = get_Twc(cam_id)
            fx, fy = rect_info_msgs[cam_id].K[0], rect_info_msgs[cam_id].K[4]

            # t0
            rect_rgb_msg, rect_depth_msg = rectify(rgb_msg, depth_msg, mx, my, bridge)
            edge_resp = predict_edge(rect_depth_msg, fx, fy)
            obb_resp = compute_obb(rect_depth_msg, rect_rgb_msg, edge_resp.mask,
                    Twc, std_msgs.msg.String(cam_id))
            # t1
            evaluators[cam_id].put(obb_resp.output)
            
            # TODO remove below visualizer..
            cvrgb = np.frombuffer(rect_rgb_msg.data, dtype=np.uint8).reshape(rect_rgb_msg.height, rect_rgb_msg.width, 3)
            cvedge = np.frombuffer(edge_resp.mask.data, dtype=np.uint8).reshape(edge_resp.mask.height, edge_resp.mask.width, 2)
            cv2.imshow("rgb", cvrgb)
            cv2.imshow("edge", (cvedge[:,:,0]==1).astype(np.uint8)*255 )
            c = cv2.waitKey(1000)
            if c == ord('q'):
                exit(1)
