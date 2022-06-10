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
from cv_bridge import CvBridge #< doesn't work for python3
import subprocess
from util import *
import unet_ext
import pickle
from os import makedirs

def get_topic(filename, topic):
    bag = rosbag.Bag(filename)
    messages = []
    for topic, msg, t in bag.read_messages(topics=[topic]):
        messages.append(msg)
    print("len(%s) = %d" % (topic, len(messages))  )
    return messages

def get_base(fullfn):
    return osp.splitext(osp.basename(fullfn))[0]

def ParseMarker(cv_gt, rgb=None):
    la = np.logical_and
    lo = np.logical_or

    # Get Red, Green, Blue dot
    # Get Yellow edges
    reddots = la(la(cv_gt[:,:,0]==0,cv_gt[:,:,1]==0),cv_gt[:,:,2]==255)
    bluedots = la(la(cv_gt[:,:,0]==255,cv_gt[:,:,1]==0),cv_gt[:,:,2]==0)
    yellowedges = la(la(cv_gt[:,:,0]==0,cv_gt[:,:,1]==255),cv_gt[:,:,2]==255)

    outline = la(la(cv_gt[:,:,0]==255,cv_gt[:,:,1]==255),cv_gt[:,:,2]==255)
    for c in [reddots, bluedots]:
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
    for element, mask_of_dots in {'o':reddots,'p':bluedots}.items():
        _,_,_, centroids = cv2.connectedComponentsWithStats(mask_of_dots.astype(np.uint8))
        for pt in centroids[1:,:]: # 1st row : centroid of 'zero background'
            c, r = int(pt[0]), int(pt[1])
            pidx = plane_marker0[r,c]
            if pidx <= 0:
                import pdb; pdb.set_trace()
            assert(pidx > 0)
            if not pidx in vertices:
                vertices[pidx] = {}
            vertices[pidx][element] = pt

    planemarker2vertices = []
    for pidx in range(1,n_planesegments): # 0 for edge and background
        valid = False
        cp = plane_centroids[pidx,:].astype(np.int32)
        x0,y0,w,h,area = plane_stats[pidx,:]
        if pidx in vertices:
            keys = vertices[pidx].keys()
            c1 = min(w,h) > 5
            c2 = plane_marker0[cp[1],cp[0]] == pidx
            c3 = 'o' in keys and len(keys) > 1
            if c1 and c2 and c3:
                valid = True
        if not valid:
            plane_marker[plane_marker0==pidx] = 0
            continue
        arr_oyz = - np.ones((4,),np.float32)
        arr_oyz[:2] =  vertices[pidx]['o']
        arr_oyz[2:4] =  vertices[pidx]['p']
        planemarker2vertices.append( [pidx, arr_oyz, cp] )

    dist = cv2.distanceTransform( (~outline).astype(np.uint8),
            distanceType=cv2.DIST_L2, maskSize=5)
    n_inssegments, marker0, _, _ \
            = cv2.connectedComponentsWithStats((dist>5).astype(np.uint8))

    ## thick outline
    #outline = dist < 3.
    
    # Sync correspondence of marker and  plane_marker
    marker = np.zeros_like(marker0)
    front_marker = np.zeros_like(marker0)
    for pidx, _, cp in planemarker2vertices:
        idx = marker0[cp[1],cp[0]]
        marker[marker0==idx] = pidx
        front_marker[plane_marker0==pidx] = pidx

    verbose=False
    if verbose:
        cv2.imshow("outline", 255*outline.astype(np.uint8))
        cv2.imshow("color_pm0", color_pm0)
        cv2.imshow("marker", GetColoredLabel(marker))
        cv2.imshow("front_marker", GetColoredLabel(front_marker))
        dst = GetColoredLabel(marker)
        for pidx, arr_oyz, cp in planemarker2vertices:
            pt_org = tuple(arr_oyz[:2].astype(np.int32).tolist())
            pt_y = tuple(arr_oyz[2:4].astype(np.int32).tolist())
            cv2.line(dst,pt_org,pt_y, (100,255,100),2)
            cv2.circle(dst,pt_org,3,(0,0,255),-1)
        if rgb is not None:
            dst = cv2.addWeighted(dst, 0.4, rgb, 0.6, 0.)
        cv2.imshow("dst", dst)
        if ord('q') == cv2.waitKey():
            exit(1)
    return outline, marker, front_marker, planemarker2vertices

def ParseGroundTruth(cv_gt, rgb, depth, K, D, fn_rosbag, max_depth):
    # 1) watershed, 꼭지점 따기.
    # 2) OBB - Roll angle 따기.
    # 3) unet_ext : Unprojection,
    # 4) unet_ext : Euclidean Cluster+oBB?
    outline, marker, front_marker, planemarker2vertices = ParseMarker(cv_gt, rgb)

    # mask2obb, GetUV와 같은 normalization map.
    nr, nc = marker.shape
    nu_map = np.zeros((nr,nc),dtype=np.float)
    nv_map = np.zeros((nr,nc),dtype=np.float)
    for r in range(nr):
        for c in range(nc):
            if D is None:
                u,v = float(c), float(r)
                nu_map[r,c] = (u-K[0,2])/K[0,0]
                nv_map[r,c] = (v-K[1,2])/K[1,1]
            else:
                pt = np.array((c,r),np.float).reshape((1,1,2))
                nuv = cv2.undistortPoints(pt, K, D)
                nu_map[r,c] = nuv[0,0,0]
                nv_map[r,c] = nuv[0,0,1]

    obb_tuples = unet_ext.ComputeOBB(front_marker, marker, planemarker2vertices, depth,
            nu_map, nv_map, max_depth)
    obbs = []
    for idx, pose, scale in obb_tuples:
        #  pose = (x,y,z, qw,qx,qy,qz) for transform {camera} <- {box}
        obbs.append( {'id':idx, 'pose':pose, 'scale':scale } )

    init_floormask = GetInitFloorMask(cv_gt)
    return obbs, init_floormask, marker

def GetInitFloorMask(cv_gt):
    la = np.logical_and
    lo = np.logical_or

    green_area = la(la(cv_gt[:,:,0]==0,cv_gt[:,:,1]==255),cv_gt[:,:,2]==0)
    n_component, _, stats, _ = cv2.connectedComponentsWithStats(green_area.astype(np.uint8))
    bridge = CvBridge()
    init_floormask = bridge.cv2_to_imgmsg(255*green_area.astype(np.uint8), encoding="8UC1")
    if stats.shape[0] < 2:
        init_floormask = None
    else:
        for j in range(1,stats.shape[0]):
            if stats[j,1] < 10:
                init_floormask = None
                break
    return init_floormask

def make_dataset_dir(name='obb_dataset'):
    script_fn = osp.abspath(__file__)
    pkg_dir = str('/').join(script_fn.split('/')[:-3])
    output_path = osp.join(pkg_dir, name)
    exist_files = set()
    if not osp.exists(output_path):
        makedirs(output_path)
    else:
        exist_labels = glob2.glob(osp.join(output_path,'*.png'),recursive=True)
        for each in exist_labels:
            #if each == '/home/geo/ws/dnn_experiment/ros_unet/obb_dataset_test/helios_dist_2022-05-20-12-10-26_cam0.png':
            #    import pdb; pdb.set_trace()
            groups = re.findall("(.*)_(cam0|cam1).png", each)
            if len(groups) != 1:
                import pdb; pdb.set_trace()
            rosbagfn, cam_id = groups[0]
            exist_files.add(get_base(rosbagfn))
    return output_path, exist_files
