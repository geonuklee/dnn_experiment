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

from sensor_msgs.msg import Image, CameraInfo, Imu
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
    if D.sum() == 0.:
        return camera_info, None, None
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
    if mx is None:
        rect_rgb, rect_depth = rgb, depth
    else:
        rect_rgb = cv2.remap(rgb,mx,my,cv2.INTER_NEAREST)
        rect_depth = cv2.remap(depth,mx,my,cv2.INTER_NEAREST)
    rect_rgb_msg = bridge.cv2_to_imgmsg(rect_rgb,encoding='bgr8')
    rect_depth_msg = bridge.cv2_to_imgmsg(rect_depth,encoding='32FC1')
    assert(rgb_msg.header.frame_id == depth_msg.header.frame_id)
    rect_rgb_msg.header.frame_id = rect_depth_msg.header.frame_id = rgb_msg.header.frame_id
    return rect_rgb_msg, rect_depth_msg, rect_depth, rect_rgb

#def get_Twc(cam_id):
#    rate = rospy.Rate(2)
#    while True:
#        Twc = rospy.get_param("/%s/base_T_cam"%cam_id, None)
#        if Twc is not None:
#            break
#        rate.sleep()
#    pose = convert2pose(Twc)
#    return pose

class Sub:
    def __init__(self, rgb, depth, info, imu):
        self.sub_depth = rospy.Subscriber(depth, Image, self.cb_depth, queue_size=1)
        self.sub_rgb   = rospy.Subscriber(rgb, Image, self.cb_rgb, queue_size=1)
        self.sub_info  = rospy.Subscriber(info, CameraInfo, self.cb_info, queue_size=1)
        self.sub_imu   = rospy.Subscriber(imu, Imu, self.cb_imu, queue_size=1)
        self.depth = None
        self.rgb = None
        self.info = None
        self.imu = None

    def cb_imu(self, msg):
        self.imu = msg

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
    if not osp.exists(pick_fn):
        return None
    f = open(pick_fn,'r')
    pick = pickle.load(f)
    f.close()
    return pick

if __name__=="__main__":
    rospy.init_node('~', anonymous=True)
    rospy.wait_for_service('~PredictEdge')
    predict_edge = rospy.ServiceProxy('~PredictEdge', ros_unet.srv.ComputeEdge)
    pub_eval = rospy.Publisher('~eval', Image,queue_size=1)


    rospy.wait_for_service('~ComputeObb')
    compute_obb = rospy.ServiceProxy('~ComputeObb', ros_unet.srv.ComputeObb)

    rospy.wait_for_service('~GetBg')
    get_bg = rospy.ServiceProxy('~GetBg', ros_unet.srv.GetBg)

    #rospy.wait_for_service('~Cgal/ComputeObb')
    #cgal_compute_obb = rospy.ServiceProxy('~Cgal/ComputeObb', ros_unet.srv.ComputePoints2Obb)
    #rospy.wait_for_service('~Ransac/ComputeObb')
    #ransac_compute_obb = rospy.ServiceProxy('~Ransac/ComputeObb', ros_unet.srv.ComputePoints2Obb)

    do_eval = rospy.get_param("~do_eval")=='true'
    if do_eval:
        rosbagfn = rospy.get_param("~filename")
        datasetnames = rospy.get_param("~datasetnames").split(",")
        for datasetname in datasetnames:
            pick = get_pick_fromrosbag(datasetname, rosbagfn)
            if pick is None:
                continue
            else:
                break
        if pick is None:
            rospy.logerr("Can't find %s in [%s]"%(rosbagfn,datasetnames) )
            import pdb; pdb.set_trace()
            exit(1)

    bridge = CvBridge()
    cam_id = "cam0"
    sub = Sub("~%s/rgb"%cam_id, "~%s/depth"%cam_id, "~%s/info"%cam_id, "~%s/imu"%cam_id)
    rect_info_msgs = {}

    rate = rospy.Rate(hz=30)
    while not rospy.is_shutdown():
        if sub.info is None:
            continue
        rect_info_msgs[cam_id], mx, my = get_rectification(sub.info)
        break

    while not rospy.is_shutdown():
        if sub.rgb is None or sub.depth is None or sub.info is None or sub.imu is None:
            continue
        # TODO Hard code for test with old data
        if sub.rgb.header.frame_id == 'arena_camera':
            sub.info.header.frame_id = sub.rgb.header.frame_id = sub.depth.header.frame_id = 'cam0_arena_camera'

        rect_info_msg = rect_info_msgs[cam_id]
        fx, fy = rect_info_msg.K[0], rect_info_msg.K[4]
        rect_K = np.array(rect_info_msgs[cam_id].K,np.float).reshape((3,3))
        rect_D = np.array(rect_info_msgs[cam_id].D,np.float).reshape((-1,))
        rect_rgb_msg, rect_depth_msg, rect_depth, rect_rgb = rectify(sub.rgb, sub.depth, mx, my, bridge)

        '''
        foolr mask 대신 GetBg로 처리.
        1) floor, too far에 있는 OBB는 필터링.
        2) wall을 이루는 OBB의 숫자, 크기, 평행 여부, 면적 비율 비교로 wall 여부 판정.
        '''
        bg_res = get_bg(rect_rgb_msg,rect_depth_msg,rect_info_msg,sub.imu)
        #p_marker = bridge.imgmsg_to_cv2(bg_res.p_marker, desired_encoding='bgr8')
        #p_marker = bridge.imgmsg_to_cv2(bg_res.p_marker, desired_encoding='passthrough')
        #p_mask = bridge.imgmsg_to_cv2(bg_res.p_mask, desired_encoding='passthrough')
        #rect_depth = bridge.imgmsg_to_cv2(rect_depth_msg, desired_encoding='passthrough').copy()
        #rect_depth[p_mask>0] = 0. # p_mask 1 for floor, 2 for wall
        #rect_depth_msg = bridge.cv2_to_imgmsg(rect_depth,encoding='32FC1')
        t0 = time.time()
        edge_resp = predict_edge(rect_rgb_msg, rect_depth_msg, fx, fy)
        obb_resp = compute_obb(rect_depth_msg, rect_rgb_msg, edge_resp.edge, rect_info_msg,
                std_msgs.msg.String(cam_id), bg_res.p_mask)
        t1 = time.time()

        if do_eval:
            outputlist, dst = Evaluate2D(obb_resp, pick['marker'], rect_rgb)
            dst_msg = bridge.cv2_to_imgmsg(dst,encoding='bgr8')
            pub_eval.publish(dst_msg)

        #print("etime = ", t1-t0)
        rate.sleep()

