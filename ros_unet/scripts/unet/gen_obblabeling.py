#!/usr/bin/python2
#-*- coding:utf-8 -*-

import rospy
import rosbag
import os
import re
import numpy as np
import cv2
import glob2 # For recursive glob for python2
from os import path as osp
from cv_bridge import CvBridge
import subprocess
from util import *
import unet_ext

def get_topic(filename, topic):
    bag = rosbag.Bag(filename)
    messages = []
    for topic, msg, t in bag.read_messages(topics=[topic]):
        messages.append(msg)
    print("len(%s) = %d" % (topic, len(messages))  )
    return messages

def ParseRosbag(fullfn):
    fn = osp.basename(fullfn)
    command  = "rosbag info %s"%fullfn
    command += "| grep image_raw"
    infos = os.popen(command).read()
    depth_groups = re.findall("\ (\/(.*)\/(k4a|helios2)\/(depth|depth_to_rgb)\/image_raw)\ ", infos)
    rgb_groups = re.findall("\ (\/(.*)\/(k4a|aligned)\/(rgb|rgb_to_depth)\/image_raw)\ ", infos)
    rgbs = {}
    bridge = CvBridge()
    dsize = (1280,960)

    for topic, cam_id, _, rgb_type in rgb_groups:
        if cam_id not in rgbs:
            rgbs[cam_id] = {}
        rgbs[cam_id][rgb_type] = topic

    for topic, cam_id, camera_type, depth_type in depth_groups:
        depth_messages = get_topic(fullfn, topic)
        depth0 = bridge.imgmsg_to_cv2(depth_messages[0], desired_encoding="32FC1")

        if depth0.shape[1] < 600: # Too small image
            continue
        if camera_type == "k4a": # Too poor
            continue
        if depth_type == 'depth_to_rgb':
            rgb_topic = rgbs[cam_id]['rgb']
        else:
            rgb_topic = rgbs[cam_id]['rgb_to_depth']
        rgb_messages = get_topic(fullfn, rgb_topic)
        ros_info = get_topic(fullfn, "/%s/helios2/camera_info"%cam_id)[0]
        K = np.array( ros_info.K ,dtype=np.float).reshape((3,3))
        D = np.array( ros_info.D, dtype=np.float).reshape((-1,))
        info = {"K":K, "D":D, "width":ros_info.width, "height":ros_info.height}
        b_pass = False
        while True:
            c = 0
            for i, msg in enumerate(rgb_messages):
                #print(i)
                orgb = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8" )
                rgb  = cv2.resize(orgb, dsize)
                cv2.imshow("rgb", rgb)
                c = cv2.waitKey(5)
                c = ord('a')
                if c==ord('q'):
                    exit(1)
                elif c == ord('c'):
                    b_pass = True
                    break
                elif c != 255:
                    break
            if c != 255:
                break
        if b_pass:
            continue
        #minirgb  = cv2.resize(orgb, (nr,nc))
        gray = cv2.cvtColor(orgb,cv2.COLOR_BGR2GRAY)/2
        depth0 = get_meterdepth(depth0)
        input_stack, grad, hessian, outline, convex_edge\
            = ConvertDepth2input(depth0,fx=K[0,0],fy=K[1,1])
        nr,nc = depth0.shape[0], depth0.shape[1]
        #dst  = np.zeros((nr,nc+gray.shape[1],3), np.uint8)
        #dst[:,:,:] = 50
        #dst[:nr,:nc,:][outline>0,:]=150
        #dst[:nr,:nc,:][convex_edge>0,:]=100
        #dst[gray.shape[0]:,nc:,:] = 100
        #dst[:gray.shape[0],nc:,:] = np.stack((gray,gray,gray),axis=2)
        #dst = np.vstack((dst,dst))
        dst = np.stack((gray,gray,gray),axis=2)

        cv2.imwrite("tmp.png", dst)
        callout = subprocess.call(['kolourpaint', "tmp.png"] )
        #TODO Parsing!
        # Boundary,In -Black,White ,  x - R, y - G, org - B
        #ParseGroundTruth(cv_gt=dst,shape=(nr,nc))
        break
    return

def GetScene(fn_rosbag):
    fn = osp.basename(fn_rosbag)
    command  = "rosbag info %s"%fn_rosbag
    command += "| grep image_raw"
    infos = os.popen(command).read()
    depth_groups = re.findall("\ (\/(.*)\/(k4a|helios2)\/(depth|depth_to_rgb)\/image_raw)\ ", infos)
    rgb_groups = re.findall("\ (\/(.*)\/(k4a|aligned)\/(rgb|rgb_to_depth)\/image_raw)\ ", infos)
    rgbs = {}
    bridge = CvBridge()
    for topic, cam_id, _, rgb_type in rgb_groups:
        if cam_id not in rgbs:
            rgbs[cam_id] = {}
        rgbs[cam_id][rgb_type] = topic

    for topic, cam_id, camera_type, depth_type in depth_groups:
        depth_messages = get_topic(fn_rosbag, topic)
        depth0 = bridge.imgmsg_to_cv2(depth_messages[0], desired_encoding="32FC1")

        if depth0.shape[1] < 600: # Too small image
            continue
        if camera_type == "k4a": # Too poor
            continue
        if depth_type == 'depth_to_rgb':
            rgb_topic = rgbs[cam_id]['rgb']
        else:
            rgb_topic = rgbs[cam_id]['rgb_to_depth']
        rgb_messages = get_topic(fn_rosbag, rgb_topic)
        ros_info = get_topic(fn_rosbag, "/%s/helios2/camera_info"%cam_id)[0]
        K = np.array( ros_info.K ,dtype=np.float).reshape((3,3))
        D = np.array( ros_info.D, dtype=np.float).reshape((-1,))
        info = {"K":K, "D":D, "width":ros_info.width, "height":ros_info.height}
        for i, msg in enumerate(rgb_messages):
            orgb = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8" )
            break
        break
    return orgb, depth0, info


def ParseGroundTruth(cv_gt, fn_rosbag):
    # 1) watershed, 꼭지점 따기.
    # 2) OBB - Roll angle 따기.
    # 3) cpp_ext : Unprojection,
    # 4) cpp_ext : Euclidean Cluster+oBB?
    la = np.logical_and
    lo = np.logical_or

    # Get Red, Green, Blue dot
    # Get Yellow edges
    reddots = la(la(cv_gt[:,:,0]==0,cv_gt[:,:,1]==0),cv_gt[:,:,2]==255)
    greendots = la(la(cv_gt[:,:,0]==0,cv_gt[:,:,1]==255),cv_gt[:,:,2]==0)
    bluedots = la(la(cv_gt[:,:,0]==255,cv_gt[:,:,1]==0),cv_gt[:,:,2]==0)
    yellowedges = la(la(cv_gt[:,:,0]==0,cv_gt[:,:,1]==255),cv_gt[:,:,2]==255)

    outline = la(la(cv_gt[:,:,0]==255,cv_gt[:,:,1]==255),cv_gt[:,:,2]==255)
    for c in [reddots, greendots, bluedots]:
        outline = la(outline, ~c)

    # outline에 red, green, blue dot 추가. 안그러면 component가 끊겨서..
    boundary = outline.copy()
    for c in [yellowedges]:
        boundary = lo(boundary, c)

    dist = cv2.distanceTransform( (~boundary).astype(np.uint8),
            distanceType=cv2.DIST_L2, maskSize=5)
    n_planesegments, plane_marker0, plane_stats, plane_centroids \
            = cv2.connectedComponentsWithStats((dist>1).astype(np.uint8) )

    color_pm0 = GetColoredLabel(plane_marker0)
    plane_marker = plane_marker0.copy()
    vertices = {}
    for element, mask_of_dots in {'o':reddots,'x':greendots,'y':bluedots}.items():
        _,_,_, centroids = cv2.connectedComponentsWithStats(mask_of_dots.astype(np.uint8))
        for pt in centroids[1:,:]: # 1st row : centroid of 'zero background'
            c, r = int(pt[0]), int(pt[1])
            pidx = plane_marker0[r,c]
            assert(pidx > 0)
            if not vertices.has_key(pidx):
                vertices[pidx] = {}
            vertices[pidx][element] = pt

    planemarker2vertices = []
    for pidx in range(1,n_planesegments): # 0 for edge and background
        valid = False
        cp = plane_centroids[pidx,:].astype(np.int32)
        x0,y0,w,h,area = plane_stats[pidx,:]
        if vertices.has_key(pidx):
            keys = vertices[pidx].keys()
            c1 = min(w,h) > 5
            c2 = plane_marker0[cp[1],cp[0]] == pidx
            c3 = 'o' in keys and len(keys) > 1
            if c1 and c2 and c3:
                valid = True
        if not valid:
            plane_marker[plane_marker0==pidx] = 0
            continue
        arr_oxy = - np.ones((6,),np.float32)
        arr_oxy[:2] =  vertices[pidx]['o']
        if 'x' in keys:
            arr_oxy[2:4] =  vertices[pidx]['x']
        elif 'y' in keys:
            arr_oxy[4:] =  vertices[pidx]['y']
        planemarker2vertices.append( (pidx, arr_oxy, cp) )

    dist = cv2.distanceTransform( (~outline).astype(np.uint8),
            distanceType=cv2.DIST_L2, maskSize=5)
    n_inssegments, marker0, _, _ \
            = cv2.connectedComponentsWithStats((dist>1).astype(np.uint8))
    
    # Sync correspondence of marker and  plane_marker
    marker = np.zeros_like(marker0)
    front_plane = np.zeros_like(marker0)
    for pidx, _, cp in planemarker2vertices:
        idx = marker0[cp[1],cp[0]]
        marker[marker0==idx] = pidx
        front_plane[plane_marker0==pidx] = pidx

    rgb, depth, info = GetScene(fn_rosbag)

    verbose=True
    if verbose:
        cv2.imshow("color_pm0", color_pm0)
        cv2.imshow("marker", GetColoredLabel(marker))
        cv2.imshow("front_plane", GetColoredLabel(front_plane))
        dst = GetColoredLabel(marker)
        for pidx, arr_oxy, cp in planemarker2vertices:
            pt_org = tuple(arr_oxy[:2].astype(np.int32).tolist())
            if arr_oxy[2] > 0:
                pt_x = tuple(arr_oxy[2:4].astype(np.int32).tolist())
                cv2.line(dst,pt_org,pt_x, (255,255,255),2)
            if arr_oxy[4] > 0:
                pt_y = tuple(arr_oxy[4:].astype(np.int32).tolist())
                cv2.line(dst,pt_org,pt_y, (255,255,255),2)
            cv2.circle(dst,pt_org,3,(0,0,255),-1)
            #cv2.circle(dst,(cp[0],cp[1]),5,(150,150,150),2)
        cv2.imshow("dst", dst)
        cv2.waitKey()

    return


if __name__ == '__main__':
    if True:
        # tmp1.png
        #ParseRosbag('/home/geo/dataset/unloading/stc2021/stc_2021-09-02-10-28-32.bag')
        # tmp2.png
        #ParseRosbag('/home/geo/dataset/unloading/stc2021/stc_2021-08-19-17-10-55.bag')

        # TODO OBB from, above cv_gt
        #cv_gt = cv2.imread("tmp.png")[:480,:640,:]
        #ParseGroundTruth(cv_gt,(480,640))

        cv_gt = cv2.imread("tmp1.png")
        fn_rosbag = "/home/geo/dataset/unloading/stc2021/stc_2021-09-02-10-28-32.bag"
        ParseGroundTruth(cv_gt, fn_rosbag)
        #cv_gt = cv2.imread("tmp2.png")
        #fn_rosbag = "/home/geo/dataset/unloading/stc2021/stc_2021-08-19-17-10-55.bag"
        #ParseGroundTruth(cv_gt, fn_rosbag)


    else:
        rosbag_path = '/home/geo/dataset/unloading/**/*.bag' # remove hardcoding .. 
        rosbagfiles = glob2.glob(rosbag_path,recursive=True)
        dsize = (1280,960)
        script_fn = osp.abspath(__file__)
        pkg_dir = str('/').join(script_fn.split('/')[:-3])
        dataset_path = osp.join(pkg_dir, 'obb_dataset')
        for fullfn in rosbagfiles:
            ParseRosbag(fullfn)
            break

