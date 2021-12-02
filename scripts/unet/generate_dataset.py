#!/usr/bin/python2
#-*- coding:utf-8 -*-

import rospy
import rosbag
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import glob2 # For recursive glob for python2
from os import path as osp
import os
import re
import numpy as np

#import ros_numpy # ros_numpy doesn't support numpify for compressed rosbag topic
from cv_bridge import CvBridge
from os import makedirs

import subprocess

'''
TODO : Torch Dataset from generated files

'''

def get_topic(filename, topic):
    bag = rosbag.Bag(filename)
    msg = None
    for topic, msg, t in bag.read_messages(topics=[topic]):
        msg = msg
        break
    return msg

if __name__ == '__main__':
    rosbag_path = '/home/geo/dataset/unloading/**/*.bag' # remove hardcoding .. 
    rosbagfiles = glob2.glob(rosbag_path,recursive=True)
    dsize = (1280,960)

    script_fn = osp.abspath(__file__)
    pkg_dir = str('/').join(script_fn.split('/')[:-3])
    dataset_path = osp.join(pkg_dir, 'segment_dataset')
    print(dataset_path) 

    usages = {'train':0.9, 'valid':0.1}
    n_frames = {}
    if not osp.exists(dataset_path):
        makedirs(dataset_path)
        for usage in usages.keys():
            usagepath = osp.join(dataset_path,usage)
            makedirs(usagepath)
            n_frames[usage] = 0
    else:
        # Count file list
        for usage in usages.keys():
            depth_files = osp.join(dataset_path,usage,'*_depth.npy')
            n_frame = len(glob2.glob(depth_files))
            print(n_frame)
            n_frames[usage] = n_frame

    bridge = CvBridge()
    fn_rosbag_list = osp.join(dataset_path,'rosbag_list.txt')
    if osp.exists(fn_rosbag_list):
        fp_rosbaglist = open(fn_rosbag_list,'r')
        closed_set = fp_rosbaglist.read().split("\n")
        fp_rosbaglist.close()
        closed_set = filter(lambda fn: fn != '', closed_set)
        closed_set = set(closed_set)
    else:
        closed_set = set()

    fp_rosbaglist = open(fn_rosbag_list,'a')

    for fullfn in rosbagfiles:
        fn = osp.basename(fullfn)
        if fn in closed_set:
            continue
        closed_set.add(fn)
        command  = "rosbag info %s"%fullfn
        command += "| grep image_raw"
        infos = os.popen(command).read()
        depth_groups = re.findall("\ (\/(.*)\/(k4a|helios2)\/(depth|depth_to_rgb)\/image_raw)\ ", infos)
        rgb_groups = re.findall("\ (\/(.*)\/(k4a|aligned)\/(rgb|rgb_to_depth)\/image_raw)\ ", infos)
        rgbs = {}
        for topic, cam_id, _, rgb_type in rgb_groups:
            if cam_id not in rgbs:
                rgbs[cam_id] = {}
            rgbs[cam_id][rgb_type] = topic

        b_results_from_fn = False
        for topic, cam_id, camera_type, depth_type in depth_groups:
            depth_msg = get_topic(fullfn, topic)
            depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            if depth.shape[1] < 600: # Too small image
                continue
            if depth.max() > 100.: # Convert [mm] to [m]
                depth /= 1000.
            if camera_type == "k4a": # Too poor
                continue
            if depth_type == 'depth_to_rgb':
                rgb_topic = rgbs[cam_id]['rgb']
            else:
                rgb_topic = rgbs[cam_id]['rgb_to_depth']
            rgb_msg = get_topic(fullfn, rgb_topic)
            orgb = bridge.imgmsg_to_cv2(rgb_msg,desired_encoding="bgr8" )
            rgb  = cv2.resize(orgb, dsize)
            minirgb  = cv2.resize(orgb, (400,400) )

            lap3 = cv2.Laplacian(depth, cv2.CV_32FC1, ksize=3)
            lap3 = cv2.resize(lap3, dsize)
            lap5 = cv2.Laplacian(depth, cv2.CV_32FC1, ksize=5)
            lap5 = cv2.resize(lap5, dsize)
            depth = cv2.resize(depth,dsize,interpolation=cv2.INTER_NEAREST)

            blap3 = (lap3<-0.01).astype(np.uint8)*255
            blap5 = (lap5<-0.1 ).astype(np.uint8)*255

            #print(topic, cam_id, camera_type, depth_type)
            if n_frames['train'] < usages['train']*len(closed_set) :
                usage = 'train'
            else:
                usage = 'valid'

            usagepath = osp.join(dataset_path,usage)
            write_format = usagepath+'/%d_%s.%s'
            fp_info = open(write_format%(n_frames[usage],"info","txt"), 'w')
            line = '\t'.join([fn, topic, cam_id, camera_type, depth_type])

            fp_info.write(line+"\n")
            fp_info.flush()

            dst = np.zeros((blap5.shape[0], blap5.shape[1]+minirgb.shape[1],3), np.uint8)
            for i in range(3):
                dst[:,:blap5.shape[1],i] = blap5
                dst[minirgb.shape[0]:,blap5.shape[1]:,i] = 100
            dst[:minirgb.shape[0],blap5.shape[1]:,:] = minirgb

            cv2.imshow("rgb", rgb)
            cv2.imshow("dst", dst)
            c = cv2.waitKey()
            if c == ord('q'):
                exit(1)
            elif c == ord('c'):
                continue

            fn_truth = osp.join(usagepath,'%d_gt.png'%n_frames[usage])
            cv2.imwrite(fn_truth, dst)
            callout = subprocess.call(['kolourpaint', fn_truth] )

            np.save(write_format%( n_frames[usage],"depth","npy"), depth)
            np.save(write_format%( n_frames[usage],"lap3" ,"npy"), lap3)
            np.save(write_format%( n_frames[usage],"lap5" ,"npy"), lap5)
            np.save(write_format%( n_frames[usage],"rgb" ,"npy"), rgb)

            n_frames[usage] += 1
            b_results_from_fn = True

        if b_results_from_fn:
            # Update rosbaglist.txt
            fp_rosbaglist.write(fn+"\n")
            fp_rosbaglist.flush()

    fp_rosbaglist.close()
