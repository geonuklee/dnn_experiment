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
from sensor_msgs.msg import Image as rosImage
import geometry_msgs
import cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as rotation_util
from evaluator import VisualizeGt, Evaluator, SceneEval # TODO change file
from ros_eval import *

from tkinter import * 
from tkinter import messagebox
from unet.gen_obblabeling import *
from unet.util import ConvertDepth2input, GetColoredLabel

def AskMakeLabelforIt(messagebox):
    output=messagebox.askyesnocancel("Re-Edit?","(y)es re-edit; (n)o next scene; (c)ancel, for quit")
    if output is None:
        exit(1)
    pass_it = not output
    #import pdb; pdb.set_trace()
    return pass_it

def MakeLabel(rect_gray, rect_depth, label_fn):
    ngray = .8*rect_gray
    dst = np.stack((ngray,ngray,ngray),axis=2)
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

def ShowObb(scen_eval, pub_marker,pub_edges,pub_planes,pub_rgb, pick):
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        scene_eval.pubGtObb()
        rgb, plane_marker,marker,outline\
                =pick['rgb'].copy(), pick['plane_marker'],pick['marker'],pick['outline']
        #import pdb; pdb.set_trace()
        dst_marker = GetColoredLabel(marker,text=True)
        rgb = cv2.addWeighted(rgb,0.4,dst_marker,0.4,1.)
        dst = np.vstack((rgb.copy(),dst_marker))
        dst_marker_msg = bridge.cv2_to_imgmsg(dst,encoding='bgr8')

        dst_edge = cv2.cvtColor(255*outline.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        dst_edge = cv2.addWeighted(rgb,0.5,dst_edge,0.5,1.)
        dst_edge_msg = bridge.cv2_to_imgmsg(dst_edge,encoding='bgr8')

        dst_plane = GetColoredLabel(plane_marker,text=True)
        dst_plane_msg = bridge.cv2_to_imgmsg(dst_plane,encoding='bgr8')

        pub_marker.publish(dst_marker_msg)
        pub_edges.publish(dst_edge_msg)
        pub_planes.publish(dst_plane_msg)
        
        rgb_msg = bridge.cv2_to_imgmsg(rgb,encoding='bgr8')
        pub_rgb.publish(rgb_msg)

        n0 = scene_eval.pub_gt_obb.get_num_connections()
        n1 = scene_eval.pub_gt_pose.get_num_connections()
        if n0 > 0 and n1 > 0:
            break
        rate.sleep()
    return

def GetPCenter(_rect_plane_marker, _rectK):
    p_marker = _rect_plane_marker.copy()
    K = _rectK.copy()
    w,h = float(p_marker.shape[1]), float(p_marker.shape[0])
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    r = round(fx/fy)
    if fx != fy:
        try:
            p_marker = cv2.resize(p_marker, (int(w), int(r*h)) )
        except:
            import pdb; pdb.set_trace()
    fy = fx
    cy = r*cy
    pindices = np.unique(p_marker)
    normalized_pcenters = {}
    for pidx in pindices:
        if pidx == 0:
            continue
        part = p_marker==pidx
        part[0,:] = part[:,0] = part[-1,:] = part[:,-1] = 0
        dist_part = cv2.distanceTransform( part.astype(np.uint8),
                distanceType=cv2.DIST_L2, maskSize=5)
        loc = np.unravel_index( np.argmax(dist_part,axis=None), p_marker.shape)
        #centers[gidx] = (loc[1],loc[0])
        normalized_pcenter  = (float(loc[1]-cx)/fx, float(loc[0]-cy)/fy)
        normalized_pcenters[pidx] = normalized_pcenter
    return normalized_pcenters


def GetMargin(_rect_gt_marker, _rectK):
    gt_marker = _rect_gt_marker.copy()
    K = _rectK.copy()
    w,h = float(gt_marker.shape[1]), float(gt_marker.shape[0])
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    r = round(fx/fy)
    if fx != fy:
        try:
            gt_marker = cv2.resize(gt_marker, (int(w), int(r*h)) )
        except:
            import pdb; pdb.set_trace()
    fy = fx
    cy = r*cy
    gindices = np.unique(gt_marker)
    minwidths, margins = {}, {}
    normalized_minwidths, normalized_margins, normalized_centers = {}, {}, {}

    bounds = np.ones((int(h),int(w)),dtype=np.uint8)
    bounds[:,0] = 0
    bounds[:,-1] = 0
    bounds[0,:] = 0
    bounds[-1,:] = 0
    bounds_dist = cv2.distanceTransform( bounds, distanceType=cv2.DIST_L2, maskSize=5)

    for gidx in gindices:
        if gidx == 0:
            continue
        part = gt_marker==gidx
        part[0,:] = part[:,0] = part[-1,:] = part[:,-1] = 0
        dist_part = cv2.distanceTransform( part.astype(np.uint8),
                distanceType=cv2.DIST_L2, maskSize=5)
        loc = np.unravel_index( np.argmax(dist_part,axis=None), gt_marker.shape)
        dy = float(min(loc[0], h-loc[0]))
        dx = float(min(loc[1], w-loc[1]))

        minwidth = dist_part[loc]
        normalized_minwidth = minwidth/fx
        normalized_center  = (float(loc[1]-cx)/fx, float(loc[0]-cy)/fy)

        #margin = min(dx, dy)
        margin = np.min( bounds_dist[part] )
        normalized_margin = margin/fx

        margins[gidx]              = margin
        minwidths[gidx]            = minwidth
        normalized_margins[gidx]   = normalized_margin
        normalized_minwidths[gidx] = normalized_minwidth
        normalized_centers[gidx]   = normalized_center
    return margins, minwidths, normalized_margins, normalized_minwidths, normalized_centers

if __name__=="__main__":
    rospy.init_node('labeling', anonymous=True)

    script_fn = osp.abspath(__file__)
    pkg_dir = str('/').join(script_fn.split('/')[:-2])
    usage = rospy.get_param('~target')
    dataset_name = 'obb_dataset_%s'%usage
    obbdatasetpath = osp.join(pkg_dir,dataset_name)
    output_path, exist_labels = make_dataset_dir(obbdatasetpath)
    exist_picks = glob2.glob(osp.join(output_path,'*.pick'),recursive=True)

    #rosbag_path = '/home/geo/catkin_ws/src/ros_unet/rosbag_%s/**/*.bag'%usage
    rosbag_path = osp.join(pkg_dir,'rosbag_%s/**/*.bag'%usage)
    rosbagfiles = glob2.glob(rosbag_path,recursive=True)

    cam_id = 0

    #rospy.wait_for_service('~FloorDetector/SetCamera')
    #floordetector_set_camera = rospy.ServiceProxy('~FloorDetector/SetCamera', ros_unet.srv.SetCamera)
    #rospy.wait_for_service('~FloorDetector/ComputeFloor')
    #compute_floor = rospy.ServiceProxy('~FloorDetector/ComputeFloor', ros_unet.srv.ComputeFloor)
    pub_marker = rospy.Publisher("~marker", rosImage, queue_size=1)
    pub_edges = rospy.Publisher("~edges", rosImage, queue_size=1)
    pub_planes = rospy.Publisher("~planes", rosImage, queue_size=1)
    pub_rgb = rospy.Publisher("~rgb", rosImage, queue_size=1)
    bridge = CvBridge()
    rate = rospy.Rate(10)

    root = Tk()
    root.geometry("30x20")
    add_newlabel = 'yes'==messagebox.askquestion("askquestion", "Add new label?(or check exist label)")
    y0, max_z = 50, 5.,
    obb_max_depth = 20.

    if add_newlabel:
        rosbagfiles.reverse()
    for i, fullfn in enumerate(rosbagfiles):
        print("%d/%d, %s"%(i,len(rosbagfiles),fullfn) )
        if rospy.is_shutdown():
            break
        basename = get_base(fullfn) 
        is_label_exist = basename in exist_labels
        is_pick_exist  = basename in exist_picks
        #import pdb; pdb.set_trace()
        if add_newlabel == is_label_exist and is_pick_exist:
            continue
        cam_id = 'cam0'
        gt_fn = osp.join(obbdatasetpath, '%s_%s.pick'%(basename, cam_id) )

        #bag = rosbag.Bag(pick['fullfn'])
        bag = rosbag.Bag(fullfn)
        rgb_topic, depth_topic, info_topic, imu_topic = get_topicnames(fullfn, bag, given_camid=cam_id)
        intensity_topic = '/%s/helios2/intensity_rect'%cam_id

        _, rgb_msg, _ = bag.read_messages(topics=[rgb_topic]).next()
        _, depth_msg, _ = bag.read_messages(topics=[depth_topic]).next()
        _, info_msg, _= bag.read_messages(topics=[info_topic]).next()
        try:
            _, intensity_msg, _ = bag.read_messages(topics=[intensity_topic]).next()
        except:
            rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8)\
                        .reshape(rgb_msg.height, rgb_msg.width, -1)
            intensity = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            intensity_msg = bridge.cv2_to_imgmsg(intensity, encoding='mono8')
            intensity_msg.header.frame_id = rgb_msg.header.frame_id
        if len(info_msg.D) == 0:
            rect_info_msg = info_msg
            rect_rgb_msg = rgb_msg
            rect_depth_msg = depth_msg
            rect_intensity_msg = intensity_msg
        else:
            rect_info_msg, mx, my = get_rectification(info_msg)
            rect_rgb_msg, rect_depth_msg, rect_intensity_msg, rect_depth, rect_rgb, rect_intensity\
                    = rectify(rgb_msg, depth_msg, intensity_msg, mx, my, bridge)
            if rect_rgb_msg.header.frame_id == 'arena_camera':
                rect_rgb_msg.header.frame_id = 'cam0_arena_camera'
            #import pdb; pdb.set_trace()
        cam_frame_id = rect_rgb_msg.header.frame_id
        rospy.loginfo("cam_frame_id=%s"%cam_frame_id)
    
        #floordetector_set_camera(std_msgs.msg.String(cam_id), rect_info_msg)

        reedit = False
        while not rospy.is_shutdown():
            K = np.array( info_msg.K ,dtype=np.float).reshape((3,3))
            newK = np.array( rect_info_msg.K ,dtype=np.float).reshape((3,3))
            D = np.array( info_msg.D, dtype=np.float).reshape((-1,))
            cv_rect_depth = np.frombuffer(rect_depth_msg.data, dtype=np.float32)\
                    .reshape(rect_depth_msg.height, rect_depth_msg.width)
            cv_rect_rgb = np.frombuffer(rect_rgb_msg.data, dtype=np.uint8)\
                    .reshape(rect_rgb_msg.height, rect_rgb_msg.width, -1)
            cv_rect_intensity = np.frombuffer(rect_intensity_msg.data, dtype=np.uint8)\
                    .reshape(rect_rgb_msg.height, rect_rgb_msg.width)
            except_ext, ext = osp.splitext(gt_fn)
            label_fn = osp.join(dataset_name, "%s.png"%osp.basename(except_ext) )
            rosbag_fn = osp.join('rosbag_%s'%usage, osp.basename(fullfn) )
            pick = {"K":K, "D":D, "newK":newK, "depth":cv_rect_depth, "rgb":cv_rect_rgb,
                    "intensity":cv_rect_intensity,
                "rosbag_fn":rosbag_fn, "cvgt_fn":label_fn}
            label_fn = osp.join(pkg_dir,label_fn)
            
            backup = None
            if osp.exists(label_fn):
                cv_gt = cv2.imread(label_fn)
                backup = cv_gt.copy()
                try:
                    pick['obbs'], init_floormask, pick['marker'],\
                            pick['front_marker'], pick['convex_edges'], pick['outline'],\
                            (pick['plane_marker'], pick['plane2marker'], pick['plane2coeff'])\
                            = ParseGroundTruth(cv_gt, cv_rect_rgb,
                                    cv_rect_depth, pick['newK'], None, fullfn, obb_max_depth)
                except:
                    callout = subprocess.call(['kolourpaint', label_fn] )
                pick['margins'], pick['minwidths'], pick['normalized_margins'], \
                    pick['normalized_minwidths'], pick['normalized_centers']\
                        = GetMargin(pick['marker'], newK)
                pick['normalized_pcenters'] = GetPCenter(pick['plane_marker'], newK)

                try:
                    scene_eval = SceneEval(pick, max_z, cam_id, frame_id=cam_frame_id)
                except:
                    callout = subprocess.call(['kolourpaint', label_fn] )
                    #exit(1)
                    #import pdb; pdb.set_trace()
                ShowObb(scene_eval, pub_marker,pub_edges,pub_planes,pub_rgb, pick)
                print("Write pick for %s" % gt_fn)
                #print("coeff", pick['plane2coeff'])
                #print(pick['obbs'])
                with open(gt_fn, "wb" ) as f:
                    pickle.dump(pick, f, protocol=2)
            else:
                rgb = pick['rgb']
                rgb_msg = bridge.cv2_to_imgmsg(rgb,encoding='bgr8')
                pub_marker.publish(rgb_msg)
                pub_edges.publish(rgb_msg)
                pub_rgb.publish(rgb_msg)

            # 1) ask wether make label for it or not.
            if not reedit:
                if AskMakeLabelforIt(messagebox):
                    break
            reedit = False
            # 2) call kolour for this rosbag.
            cv_gt = MakeLabel(cv_rect_intensity, cv_rect_depth, label_fn)

            # Make OBB for it
            cv2.destroyAllWindows()
            pick['obbs'], init_floormask, pick['marker'],\
                    pick['front_marker'], pick['convex_edges'], pick['outline'],\
                    (pick['plane_marker'], pick['plane2marker'], pick['plane2coeff'])\
                    = ParseGroundTruth(cv_gt, cv_rect_rgb,
                            cv_rect_depth, pick['newK'], None, fullfn, obb_max_depth)
            pick['margins'], pick['minwidths'], pick['normalized_margins'],\
                pick['normalized_minwidths'], pick['normalized_centers']\
                    = GetMargin(pick['marker'], newK)
            pick['normalized_pcenters'] = GetPCenter(pick['plane_marker'], newK)

            # 3) show obb
            scene_eval = SceneEval(pick, max_z, cam_id, frame_id=cam_frame_id)
            ShowObb(scene_eval, pub_marker,pub_edges,pub_planes,pub_rgb, pick)
            with open(gt_fn, "wb" ) as f:
                pickle.dump(pick, f, protocol=2)

            if rospy.is_shutdown():
                break

            #import pdb; pdb.set_trace()
            with open(gt_fn, "wb" ) as f:
                pickle.dump(pick, f, protocol=2)

            answer = messagebox.askyesnocancel("askquestion",
                    "Complete labeling?(Save&Finish|Save&MoreEdit|Revert)")
            if answer: # Yes
                break
            elif answer is None: # Cancel for revert
                cv2.imwrite(label_fn, backup)
            else: # No for save & more edit
                reedit =True
                pass

    root.destroy()
