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
#from cv_bridge import CvBridge #< doesn't work for python3
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

def ParseGroundTruth(cv_gt, rgb, depth, K, D, fn_rosbag, max_depth):
    # 1) watershed, 꼭지점 따기.
    # 2) OBB - Roll angle 따기.
    # 3) unet_ext : Unprojection,
    # 4) unet_ext : Euclidean Cluster+oBB?
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
    for element, mask_of_dots in {'o':reddots,'y':greendots,'z':bluedots}.items():
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
        arr_oyz = - np.ones((6,),np.float32)
        arr_oyz[:2] =  vertices[pidx]['o']
        if 'y' in keys:
            arr_oyz[2:4] =  vertices[pidx]['y']
        elif 'z' in keys:
            arr_oyz[4:] =  vertices[pidx]['z']
        planemarker2vertices.append( [pidx, arr_oyz, cp] )

    dist = cv2.distanceTransform( (~outline).astype(np.uint8),
            distanceType=cv2.DIST_L2, maskSize=5)
    n_inssegments, marker0, _, _ \
            = cv2.connectedComponentsWithStats((dist>5).astype(np.uint8))
    
    # Sync correspondence of marker and  plane_marker
    marker = np.zeros_like(marker0)
    front_marker = np.zeros_like(marker0)
    for pidx, _, cp in planemarker2vertices:
        idx = marker0[cp[1],cp[0]]
        marker[marker0==idx] = pidx
        front_marker[plane_marker0==pidx] = pidx

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

    verbose=False
    if verbose:
        cv2.imshow("color_pm0", color_pm0)
        cv2.imshow("marker", GetColoredLabel(marker))
        cv2.imshow("front_marker", GetColoredLabel(front_marker))
        dst = GetColoredLabel(marker)
        for pidx, arr_oyz, cp in planemarker2vertices:
            pt_org = tuple(arr_oyz[:2].astype(np.int32).tolist())
            if arr_oyz[2] > 0:
                pt_y = tuple(arr_oyz[2:4].astype(np.int32).tolist())
                cv2.line(dst,pt_org,pt_y, (100,255,100),2)
            if arr_oyz[4] > 0:
                pt_z = tuple(arr_oyz[4:].astype(np.int32).tolist())
                cv2.line(dst,pt_org,pt_z, (100,100,255),2)
            cv2.circle(dst,pt_org,3,(0,0,255),-1)
            #cv2.circle(dst,(cp[0],cp[1]),5,(150,150,150),2)
        dst = cv2.addWeighted(dst, 0.4, rgb, 0.6, 0.)
        cv2.imshow("dst", dst)
        cv2.waitKey()

    return obbs

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
            groups = re.findall("(.*)_(cam0|cam1).png", each)
            if len(groups) != 1:
                import pdb; pdb.set_trace()
            rosbagfn, cam_id = groups[0]
            exist_files.add(get_base(rosbagfn))
    return output_path, exist_files

