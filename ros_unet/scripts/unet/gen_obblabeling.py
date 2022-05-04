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

def ParseRosbag(output_path, fullfn):
    print('Check file %s ...'%fullfn)
    fn = osp.basename(fullfn)
    command  = "rosbag info %s"%fullfn
    command += "| grep image_raw"
    infos = os.popen(command).read()
    if False:
        depth_groups = re.findall("\ (\/(.*)\/(k4a|helios2)\/(depth|depth_to_rgb)\/image_raw)\ ", infos)
        rgb_groups = re.findall("\ (\/(.*)\/(k4a|aligned)\/(rgb|rgb_to_depth)\/image_raw)\ ", infos)
    else:
        depth_groups = re.findall("\ (\/(.*)\/(helios2)\/(depth|depth_to_rgb)\/image_raw)\ ", infos)
        rgb_groups = re.findall("\ (\/(.*)\/(aligned)\/(rgb|rgb_to_depth)\/image_raw)\ ", infos)
        #depth_groups = [('/cam0/helios2/depth/image_raw', 'cam0', 'helios2', 'depth')]
        #rgb_groups = [('/cam0/aligned/rgb_to_depth/image_raw', 'cam0', 'aligned', 'rgb_to_depth')]
    rgbs = {}
    #bridge = CvBridge()
    dsize = (1280,960)

    for topic, cam_id, _, rgb_type in rgb_groups:
        if cam_id not in rgbs:
            rgbs[cam_id] = {}
        rgbs[cam_id][rgb_type] = topic

    cams = set()
    for topic, cam_id, camera_type, depth_type in depth_groups:
        if cam_id in cams:
            continue
        cams.add(cam_id)
        depth_messages = get_topic(fullfn, topic)
        #depth0 = bridge.imgmsg_to_cv2(depth_messages[0], desired_encoding="32FC1")
        depth0 = depth_messages[0]
        depth0 = np.frombuffer(depth0.data, dtype=np.float32).reshape(depth0.height, depth0.width)
        if depth0.shape[1] < 600: # Too small image
            continue
        if camera_type == "k4a": # Too poor
            continue
        if depth_type == 'depth_to_rgb':
            rgb_topic = rgbs[cam_id]['rgb']
        else:
            rgb_topic = rgbs[cam_id]['rgb_to_depth']
        rgb_messages = get_topic(fullfn, rgb_topic)
        try:
            ros_info = get_topic(fullfn, "/%s/helios2/camera_info"%cam_id)[0]
        except:
            continue
        K = np.array( ros_info.K ,dtype=np.float).reshape((3,3))
        D = np.array( ros_info.D, dtype=np.float).reshape((-1,))
        info = {"K":K, "D":D, "width":ros_info.width, "height":ros_info.height}
        b_ignore_thisfile=False
        b_break_loop=False
        while not b_break_loop:
            c = 0
            for i, msg in enumerate(rgb_messages):
                #orgb = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8" )
                orgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                dst_with_msg  = cv2.resize(orgb, dsize)
                dst_with_msg[:12,:,:] = 255
                cv2.putText(dst_with_msg, 'Do you accept this rosbag? y/n/q',
                        (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                cv2.imshow("sample", dst_with_msg)
                c = cv2.waitKey()
                if c==ord('q'):
                    exit(1)
                elif c == ord('n'):
                    b_ignore_thisfile = True
                    b_break_loop = True
                    break
                elif c == ord('y'):
                    b_break_loop = True
                    break
        if b_ignore_thisfile:
            continue
        # For each depth groups..
        nr,nc = depth0.shape[0], depth0.shape[1]
        minirgb  = cv2.resize(orgb, (nc/2,nr/2))
        gray = (0.8 * cv2.cvtColor(orgb,cv2.COLOR_BGR2GRAY)).astype(np.uint8)
        osize = (nc,nr) # width, height
       
        newK,_ = cv2.getOptimalNewCameraMatrix(K,D,osize,1)
        mx,my = cv2.initUndistortRectifyMap(K,D,None,newK,osize,cv2.CV_32F)
        dst = np.stack((gray,gray,gray),axis=2)
        dst = cv2.remap(dst,mx,my,cv2.INTER_NEAREST)

        depth0 = get_meterdepth(depth0)
        rect_depth = cv2.remap(depth0, mx,my,cv2.INTER_NEAREST)
        dst[rect_depth<0.3,:2] = 100
        rect_rgb = cv2.remap(orgb,mx,my,cv2.INTER_NEAREST)
        #input_stack, grad, hessian, outline, convex_edge\
        #    = ConvertDepth2input(depth0,fx=newK[0,0],fy=newK[1,1])
        #dst = 100*np.ones((depth0.shape[0], depth0.shape[1]+minirgb.shape[1],3), np.uint8)
        #dst[:nr,:nc,0] = 100*outline
        #dst[:nr,:nc,1] = 100*convex_edge
        #dst[:minirgb.shape[0],-minirgb.shape[1]:] = minirgb[:,:]
        #name='tmp'
        name = osp.splitext(fn)[0]
        label_fn = osp.join(output_path,"%s_%s.png"%(name,cam_id) )
        pick_fn = osp.join(output_path,"%s_%s.pick"%(name,cam_id) )
        pick = {"K":K, "D":D, "newK":newK, "depth":rect_depth, "rgb":rect_rgb,
                "fullfn":fullfn, "cvgt_fn":osp.abspath(label_fn)}
        cv2.imwrite(label_fn, dst)
        callout = subprocess.call(['kolourpaint', label_fn] )
        cv_gt = cv2.imread(label_fn)[:pick['depth'].shape[0],:pick['depth'].shape[1],:]
        #pick['obbs'] = ParseGroundTruth(cv_gt, pick['rgb'], pick['depth'], pick['newK'], None, pick['fullfn'])
        pickle.dump(pick, open(pick_fn, "wb" ))
    return

def ParseGroundTruth(cv_gt, rgb, depth, K, D, fn_rosbag):
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
            = cv2.connectedComponentsWithStats((dist>1).astype(np.uint8))
    
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

    obb_tuples = unet_ext.ComputeOBB(front_marker, marker, planemarker2vertices, depth, nu_map, nv_map)
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

if __name__ == '__main__':
    #if False:
    #    ParseRosbag('/home/geo/dataset/unloading/stc2021/stc_2021-09-02-10-28-32.bag')
    #else:
    #    # Step 2
    #    f = open('tmp1.pick','rb')
    #    pick = pickle.load(f)
    #    f.close()
    #    cv_gt = cv2.imread("tmp1.png")[:pick['depth'].shape[0],:pick['depth'].shape[1],:]
    #    obbs = ParseGroundTruth(cv_gt, pick['rgb'], pick['depth'], pick['newK'], None, pick['fullfn'])
    #    pick['obbs'] = obbs
    #    pickle.dump(pick, open('tmp1.pick','wb'))

    output_path, exist_labels = make_dataset_dir(name='obb_dataset')
    rosbag_path = '/home/geo/dataset/unloading/**/*.bag' # remove hardcoding .. 
    rosbagfiles = glob2.glob(rosbag_path,recursive=True)
    dsize = (1280,960)
    script_fn = osp.abspath(__file__)
    pkg_dir = str('/').join(script_fn.split('/')[:-3])
    output_path = osp.join(pkg_dir, 'obb_dataset')
    for fullfn in rosbagfiles:
        if get_base(fullfn) in exist_labels:
            continue
        ParseRosbag(output_path, fullfn)

