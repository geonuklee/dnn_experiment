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

#from ros_client import *
from unet.util import GetColoredLabel, Evaluate2D
from unet_ext import GetBoundary, UnprojectPointscloud

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

def convert2tf(pose):
    T = np.zeros((3,4),dtype=float)
    T[:,3] = np.array((pose.position.x, pose.position.y, pose.position.z),T.dtype).reshape((3))
    quat_xyzw = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    Rwc = rotation_util.from_quat(quat_xyzw)
    T[:3,:3] = Rwc.as_dcm().astype(T.dtype)
    return T

def convert_plane(pose, plane0):
    if plane0[0] == plane0[1] == plane0[2] == 0.:
        return plane0
    q = pose.orientation
    rot = rotation_util.from_quat( (q.x,q.y,q.z,q.w) )
    t = pose.position
    T = np.zeros((3,4), np.float32)
    T[:,:3] = rot.as_dcm()
    T[:,3] = np.array((t.x,t.y,t.z)).reshape((3,))

    nvec = np.matmul(T[:,:3], np.array(plane0[:3]).reshape((3,1)) ).reshape((3,))
    # For xz plane, y value, because assume no much roll fo camera.
    y0 = - plane0[3]/plane0[1]
    pt_a = np.matmul(T,  np.array((0,y0,0,1.),np.float).reshape((4,1)) ).reshape((3,))
    d = - nvec.dot(pt_a)

    #import pdb; pdb.set_trace()
    return (nvec[0], nvec[1], nvec[2], d)

def rectify(rgb_msg, depth_msg, mx, my, bridge):
    rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

    rect_rgb = cv2.remap(rgb,mx,my,cv2.INTER_NEAREST)
    rect_depth = cv2.remap(depth,mx,my,cv2.INTER_NEAREST)

    rect_rgb_msg = bridge.cv2_to_imgmsg(rgb,encoding='bgr8')
    rect_depth_msg = bridge.cv2_to_imgmsg(depth,encoding='32FC1')
    return rect_rgb_msg, rect_depth_msg, rect_depth, rect_rgb

def get_Twc(cam_id):
    rate = rospy.Rate(2)
    while True:
        Twc = rospy.get_param("/%s/base_T_cam"%cam_id, None)
        if Twc is not None:
            break
        rate.sleep()
    pose = convert2pose(Twc)
    return pose

def get_init_floormask(bridge, width, height, y0):
    mask = np.zeros((height,width),dtype=np.uint8)
    mask[-int(y0):,:] = 255
    init_floormask = bridge.cv2_to_imgmsg(mask, encoding="8UC1")
    return init_floormask, mask

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

def get_pkg_dir():
    return osp.abspath( osp.join(osp.dirname(__file__),'..') )

def get_pick_fromrosbag(datasetname, rosbagfn):
    base = osp.splitext( osp.basename(rosbagfn) )[0]
    pick_fn = osp.join(get_pkg_dir(), datasetname, "%s_cam0.pick"%base)
    f = open(pick_fn,'r')
    pick = pickle.load(f)
    f.close()
    return pick

if __name__=="__main__":
    rospy.init_node('~', anonymous=True)
    rospy.wait_for_service('~PredictEdge')
    predict_edge = rospy.ServiceProxy('~PredictEdge', ros_unet.srv.ComputeEdge)

    rospy.wait_for_service('~SetCamera')
    set_camera = rospy.ServiceProxy('~SetCamera', ros_unet.srv.SetCamera)

    rospy.wait_for_service('~ComputeObb')
    compute_obb = rospy.ServiceProxy('~ComputeObb', ros_unet.srv.ComputeObb)

    rospy.wait_for_service('~FloorDetector/SetCamera')
    floordetector_set_camera = rospy.ServiceProxy('~FloorDetector/SetCamera', ros_unet.srv.SetCamera)
    rospy.wait_for_service('~FloorDetector/ComputeFloor')
    compute_floor = rospy.ServiceProxy('~FloorDetector/ComputeFloor', ros_unet.srv.ComputeFloor)

    rospy.wait_for_service('~Cgal/ComputeObb')
    cgal_compute_obb = rospy.ServiceProxy('~Cgal/ComputeObb', ros_unet.srv.ComputePoints2Obb)

    rospy.wait_for_service('~Ransac/ComputeObb')
    ransac_compute_obb = rospy.ServiceProxy('~Ransac/ComputeObb', ros_unet.srv.ComputePoints2Obb)

    do_eval = rospy.get_param("~do_eval")=='true'
    if do_eval:
        rosbagfn = rospy.get_param("~filename")
        datasetname = rospy.get_param("~datasetname")
        pick = get_pick_fromrosbag(datasetname, rosbagfn)

    bridge = CvBridge()
    cam_id = "cam0"
    sub = Sub("~%s/rgb"%cam_id, "~%s/depth"%cam_id, "~%s/info"%cam_id)
    rect_info_msgs = {}
    remap_maps = {}

    rate = rospy.Rate(hz=30)
    while not rospy.is_shutdown():
        if sub.info is None:
            continue
        rect_info_msgs[cam_id], mx, my = get_rectification(sub.info)
        remap_maps[cam_id] = (mx, my)
        set_camera(std_msgs.msg.String(cam_id), rect_info_msgs[cam_id])
        floordetector_set_camera(std_msgs.msg.String(cam_id), rect_info_msgs[cam_id])
        break

    Twc = get_Twc(cam_id)
    Tfwc = convert2tf(Twc)
    while not rospy.is_shutdown():
        if sub.rgb is None or sub.depth is None or sub.info is None:
            continue
        fx, fy = rect_info_msgs[cam_id].K[0], rect_info_msgs[cam_id].K[4]
        rect_K = np.array(rect_info_msgs[cam_id].K,np.float).reshape((3,3))
        rect_D = np.array(rect_info_msgs[cam_id].D,np.float).reshape((-1,))
        rect_rgb_msg, rect_depth_msg, rect_depth, rect_rgb = rectify(sub.rgb, sub.depth, mx, my, bridge)
        init_floormask, cv_mask = get_init_floormask(bridge,rect_rgb.shape[1],rect_rgb.shape[0], y0=50)
        floor_msg = compute_floor(rect_depth_msg, rect_rgb_msg, init_floormask)
        plane_c  = floor_msg.plane
        plane_w = convert_plane(Twc, plane_c) # empty plane = no floor filter.
        #plane_w = (0.,0.,0.,1.)
        #cv2.imshow("floor", cv_mask)
        #cv2.waitKey(1)

        t0 = time.time()
        edge_resp = predict_edge(rect_rgb_msg, rect_depth_msg, fx, fy)
        obb_resp = compute_obb(rect_depth_msg, rect_rgb_msg, edge_resp.edge,
                Twc, std_msgs.msg.String(cam_id), fx, fy, plane_w)
        t1 = time.time()

        w,h = obb_resp.marker.width, obb_resp.marker.height
        marker = np.frombuffer(obb_resp.marker.data, dtype=np.int32).reshape(h,w).copy()
        boundary = GetBoundary(marker, 2)
        dist = cv2.distanceTransform( (boundary<1).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
        #marker[dist < 10.] = 0
        marker[dist < 15.] = 0

        xyzrgb, labels = UnprojectPointscloud(rect_rgb,rect_depth,marker,rect_K,rect_D,leaf_xy=0.02,leaf_z=0.01)
        A = Tfwc.astype(xyzrgb.dtype)
        B = np.hstack( (xyzrgb[:,:3], np.ones((xyzrgb.shape[0],1),xyzrgb.dtype))).T
        xyzrgb[:,:3] = np.matmul(A,B).T
        xyz = xyzrgb[:,:3].reshape(-1,).tolist()
        labels = labels.reshape(-1,).tolist()
        mvbb_resp = cgal_compute_obb(xyz,labels)
        ransac_resp = ransac_compute_obb(xyz,labels)

        if do_eval:
            eval_frame, pred_marker, dst = Evaluate2D(obb_resp, pick['marker'], rgb)
            cv2.imshow("dst", dst)
            if ord('q') == cv2.waitKey(1):
                exit(1)

        #print("etime = ", t1-t0)
        rate.sleep()

