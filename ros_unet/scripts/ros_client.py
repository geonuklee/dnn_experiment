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

from sensor_msgs.msg import Image, CameraInfo
import geometry_msgs, std_msgs
import cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as rotation_util
import time

def get_rectification(camera_info):
    K = np.array( camera_info.K ,dtype=np.float).reshape((3,3))
    D = np.array( camera_info.D, dtype=np.float).reshape((-1,))

    osize = (camera_info.width,camera_info.height)
    newK,_ = cv2.getOptimalNewCameraMatrix(K,D,osize,0)
    mx,my = cv2.initUndistortRectifyMap(K,D,None,newK,osize,cv2.CV_32F)

    new_info = CameraInfo()
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
    #rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3)
    #depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
    rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

    rgb = cv2.remap(rgb,mx,my,cv2.INTER_NEAREST)
    depth = cv2.remap(depth,mx,my,cv2.INTER_NEAREST)

    rect_rgb_msg = bridge.cv2_to_imgmsg(rgb,encoding='bgr8')
    rect_depth_msg = bridge.cv2_to_imgmsg(depth,encoding='32FC1')
    return rect_rgb_msg, rect_depth_msg

def get_Twc(cam_id):
    Twc = rospy.get_param("/%s/base_T_cam"%cam_id)
    pose = convert2pose(Twc)
    return pose

class Sub:
    def __init__(self, rgb, depth, info):
        self.sub_depth = rospy.Subscriber(depth, Image, self.cb_depth, queue_size=30)
        self.sub_rgb   = rospy.Subscriber(rgb, Image, self.cb_rgb, queue_size=30)
        self.sub_info  = rospy.Subscriber(info, CameraInfo, self.cb_info, queue_size=30)
        self.depth = None
        self.rgb = None
        self.info = None

    def cb_depth(self, msg):
        self.depth = msg

    def cb_rgb(self, msg):
        self.rgb = msg

    def cb_info(self, msg):
        self.info = msg

if __name__=="__main__":
    rospy.init_node('~', anonymous=True)
    rospy.wait_for_service('~PredictEdge')
    predict_edge = rospy.ServiceProxy('~PredictEdge', ros_unet.srv.ComputeEdge)

    rospy.wait_for_service('~SetCamera')
    set_camera = rospy.ServiceProxy('~SetCamera', ros_unet.srv.SetCamera)

    rospy.wait_for_service('~ComputeObb')
    compute_obb = rospy.ServiceProxy('~ComputeObb', ros_unet.srv.ComputeObb)
    bridge = CvBridge()

    cam_id = "cam0"
    sub = Sub("~%s/rgb"%cam_id, "~%s/depth"%cam_id, "~%s/info"%cam_id)
    rect_info_msgs = {}
    remap_maps = {}

    rate = rospy.Rate(hz=1)
    while not rospy.is_shutdown():
        if sub.info is None:
            continue
        rect_info_msgs[cam_id], mx, my = get_rectification(sub.info)
        remap_maps[cam_id] = (mx, my)
        set_camera(std_msgs.msg.String(cam_id), rect_info_msgs[cam_id])
        break

    Twc = get_Twc(cam_id)
    while not rospy.is_shutdown():
        if sub.rgb is None or sub.depth is None or sub.info is None:
            continue
        fx, fy = rect_info_msgs[cam_id].K[0], rect_info_msgs[cam_id].K[4]

        t0 = time.time()
        rect_rgb_msg, rect_depth_msg = rectify(sub.rgb, sub.depth, mx, my, bridge)
        edge_resp = predict_edge(rect_rgb_msg, rect_depth_msg, fx, fy)
        obb_resp = compute_obb(rect_depth_msg, rect_rgb_msg, edge_resp.mask,
                Twc, std_msgs.msg.String(cam_id), fx, fy)
        t1 = time.time()
        #print("etime = ", t1-t0)
        rate.sleep()

        #cvrgb = np.frombuffer(rect_rgb_msg.data, dtype=np.uint8).reshape(rect_rgb_msg.height, rect_rgb_msg.width, 3)
        #cvedge = np.frombuffer(edge_resp.mask.data, dtype=np.uint8).reshape(edge_resp.mask.height, edge_resp.mask.width, 2)
        #cv2.imshow("rgb", cvrgb)
        #cv2.imshow("edge", (cvedge[:,:,0]==1).astype(np.uint8)*255 )
        #c = cv2.waitKey(1)
        #if c == ord('q'):
        #    exit(1)
