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
from evaluator import VisualizeGt, Evaluator, SceneEval, FrameEval # TODO change file
from ros_eval import *

from tkinter import * 
from tkinter import messagebox
from unet.gen_obblabeling import *
from unet.util import ConvertDepth2input

def AskMakeLabelforIt(msg, pick_fn):
    pass_it = True
    except_ext, ext = osp.splitext(pick_fn)
    label_fn = "%s.png"%except_ext
    if osp.exists(label_fn):
        dst_with_msg = cv2.imread(label_fn)
    else:
        dst_with_msg = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    dsize = (640,480)
    dst_with_msg  = cv2.resize(dst_with_msg, dsize)
    dst_with_msg[:12,:,:] = 255
    cv2.putText(dst_with_msg, 'Do you edit label for this scene? y/n/q',
            (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    cv2.imshow("sample", dst_with_msg)
    c = cv2.waitKey()
    if c == ord('q'):
        exit(1)
    if c == 255:
        c = ord('y')
    return c != ord('y')

def MakeLabel(rect_rgb, rect_depth, label_fn):
    gray = (0.8 * cv2.cvtColor(rect_rgb,cv2.COLOR_BGR2GRAY)).astype(np.uint8)
    dst = np.stack((gray,gray,gray),axis=2)
    dst[rect_depth==0.,:2] = 0 
    #dst[depth>3.,:2] = 0 
    #fx, fy = info.K[0], info.K[4]
    #outline = ConvertDepth2input(depth, fx, fy)[3]
    #dst[outline>0, 2] = 200

    if not osp.exists(label_fn):
        cv2.imwrite(label_fn, dst)
    callout = subprocess.call(['kolourpaint', label_fn] )
    cv_gt = cv2.imread(label_fn)
    return cv_gt

def ShowObb(compute_floor,rect_depth_msg, rect_rgb_msg, y0, max_z,
        scen_eval
        ):
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        floor_msg = compute_floor(rect_depth_msg, rect_rgb_msg, y0, max_z)
        scene_eval.pubGtObb()
        n0 = scene_eval.pub_gt_obb.get_num_connections()
        n1 = scene_eval.pub_gt_pose.get_num_connections()
        if n0 > 0 and n1 > 0:
            break
        rate.sleep()
    return

if __name__=="__main__":
    pkg_dir = '/home/geo/catkin_ws/src/ros_unet' # TODO Hard coding for now
    obbdatasetpath = osp.join(pkg_dir,'obb_dataset')
    #,'*.pick') # remove hardcoding .. 
    #gt_files = glob2.glob(obbdatasetpath)
    output_path, exist_labels = make_dataset_dir(name='obb_dataset')
    rosbag_path = '/home/geo/catkin_ws/src/ros_unet/rosbag/**/*.bag'
    rosbagfiles = glob2.glob(rosbag_path,recursive=True)

    rospy.init_node('labeling', anonymous=True)
    cam_id = 0

    rospy.wait_for_service('~FloorDetector/SetCamera')
    floordetector_set_camera = rospy.ServiceProxy('~FloorDetector/SetCamera', ros_unet.srv.SetCamera)
    rospy.wait_for_service('~FloorDetector/ComputeFloor')
    compute_floor = rospy.ServiceProxy('~FloorDetector/ComputeFloor', ros_unet.srv.ComputeFloor)

    bridge = CvBridge()

    rate = rospy.Rate(10)


    root = Tk()
    root.geometry("300x200")
    add_newlabel = 'yes'==messagebox.askquestion("askquestion", "Add new label?(or check exist label)")
    #add_newlabel = False
    y0, max_z = 50, 5.,
    obb_max_depth = 5.

    rosbagfiles.reverse()
    for fullfn in rosbagfiles:
        print(fullfn)
        if rospy.is_shutdown():
            break
        basename = get_base(fullfn) 
        is_label_exist = basename in exist_labels
        if add_newlabel == is_label_exist:
            continue
        cam_id = 'cam0'
        gt_fn = osp.join(obbdatasetpath, '%s_%s.pick'%(basename, cam_id) )
        Twc = get_Twc(cam_id)

        #bag = rosbag.Bag(pick['fullfn'])
        bag = rosbag.Bag(fullfn)
        rgb_topic, depth_topic, info_topic = get_topicnames(fullfn, bag, given_camid=cam_id)

        _, rgb_msg, _ = bag.read_messages(topics=[rgb_topic]).next()
        _, depth_msg, _ = bag.read_messages(topics=[depth_topic]).next()
        _, info_msg, _= bag.read_messages(topics=[info_topic]).next()
        rect_info_msg, mx, my = get_rectification(info_msg)
        rect_rgb_msg, rect_depth_msg, rect_depth = rectify(rgb_msg, depth_msg, mx, my, bridge)
    
        floordetector_set_camera(std_msgs.msg.String(cam_id), rect_info_msg)
        floor_msg = compute_floor(rect_depth_msg, rect_rgb_msg, y0, max_z)
        plane_c = floor_msg.plane

        while not rospy.is_shutdown():
            K = np.array( info_msg.K ,dtype=np.float).reshape((3,3))
            newK = np.array( rect_info_msg.K ,dtype=np.float).reshape((3,3))
            D = np.array( info_msg.D, dtype=np.float).reshape((-1,))
            cv_rect_depth = np.frombuffer(rect_depth_msg.data, dtype=np.float32)\
                    .reshape(rect_depth_msg.height, rect_depth_msg.width)
            cv_rect_rgb = np.frombuffer(rect_rgb_msg.data, dtype=np.uint8)\
                    .reshape(rect_rgb_msg.height, rect_rgb_msg.width, -1)
            except_ext, ext = osp.splitext(gt_fn)
            label_fn = "%s.png"%except_ext

            pick = {"K":K, "D":D, "newK":newK, "depth":cv_rect_depth, "rgb":cv_rect_rgb,
                "fullfn":fullfn, "cvgt_fn":osp.abspath(label_fn)}
            
            #if osp.exists(gt_fn):
            #    pick = get_pick(gt_fn)
            #    scene_eval = SceneEval(pick, Twc, plane_c, max_z, cam_id)
            #    ShowObb(compute_floor, rect_depth_msg, rect_rgb_msg, y0, max_z, scene_eval)
            backup = None
            if osp.exists(label_fn):
                cv_gt = cv2.imread(label_fn)
                backup = cv_gt.copy()
                obbs = ParseGroundTruth(cv_gt, cv_rect_rgb, cv_rect_depth, pick['newK'], None, fullfn,
                        obb_max_depth)
                pick['obbs'] = obbs
                scene_eval = SceneEval(pick, Twc, plane_c, max_z, cam_id)
                ShowObb(compute_floor, rect_depth_msg, rect_rgb_msg, y0, max_z, scene_eval)
                print("write for %s" % gt_fn)
                with open(gt_fn, "wb" ) as f:
                    pickle.dump(pick, f)

            # 1) ask wether make label for it or not.
            pass_it = AskMakeLabelforIt(rect_rgb_msg, gt_fn)
            if pass_it:
                break

            # 2) call kolour for this rosbag.
            cv_gt = MakeLabel(cv_rect_rgb, cv_rect_depth, label_fn)

            # Make OBB for it
            cv2.destroyAllWindows()
            obbs = ParseGroundTruth(cv_gt, cv_rect_rgb, cv_rect_depth, pick['newK'], None, fullfn,
                    obb_max_depth)
            pick['obbs'] = obbs

            # 3) show obb
            scene_eval = SceneEval(pick, Twc, plane_c, max_z, cam_id)
            ShowObb(compute_floor, rect_depth_msg, rect_rgb_msg, y0, max_z, scene_eval)
            with open(gt_fn, "wb" ) as f:
                pickle.dump(pick, f)

            if rospy.is_shutdown():
                break

            #import pdb; pdb.set_trace()
            with open(gt_fn, "wb" ) as f:
                pickle.dump(pick, f)

            answer = messagebox.askyesnocancel("askquestion",
                    "Complete labeling?(Save&Finish|Save&MoreEdit|Revert)")
            if answer: # Yes
                break
            elif answer is None: # Cancel for revert
                cv2.imwrite(label_fn, backup)
            else: # No for save & more edit
                pass

    root.destroy()
