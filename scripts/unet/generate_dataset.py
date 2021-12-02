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


'''
TODO
0) From ground truth list
0) Ask user to overwrite exist frame, if it is already exist frame.
Done) Collect all /cam*/k4a/depth/image_raw, and rgb or rgb_to_depth for it.
Done) Thresholding laplacian of depth
3) Save edge - rgb. (rgb for reference)
4) Call ' kolourpaint ground_truth.png'
5) Convert above to edge.npy
6) Add 'filename' to filename list.

    script_fn = osp.abspath(__file__)
    pkg_dir = str('/').join(script_fn.split('/')[:-2])
    dataset_path = osp.join(pkg_dir, 'edge_dataset')
    usages = ['train', 'valid', 'test']
    usage_nframes = {'train':0.8*len(dataset),
        'valid':0.1*len(dataset),
        'test':0.1*len(dataset) }

    if not osp.exists(dataset_path):
        makedirs(dataset_path)
        for usage in usages:
            usagepath = osp.join(dataset_path,usage)
            if not osp.exists(usagepath):
                makedirs(usagepath)



'''

def get_topic(filename, topic):
    bag = rosbag.Bag(filename)
    msg = None
    for topic, msg, t in bag.read_messages(topics=[topic]):
        msg = msg
        break
    return msg

if __name__ == '__main__':
    # ref) https://www.notion.so/c-Python-Rosbag-API-88b49f2c413f40e1b3e8d2f09894c6bf
    rosbag_path = '/home/geo/dataset/unloading/**/*.bag' # remove hardcoding .. 
    rosbagfiles = glob2.glob(rosbag_path,recursive=True)

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
            import pdb; pdb.set_trace()

    bridge = CvBridge()
    fn_rosbag_list = osp.join(dataset_path,'rosbag_list.txt')
    if osp.exists(fn_rosbag_list):
        closed_set_fp = open(fn_rosbag_list,'r')
        closed_set = closed_set_fp.read().split("\n")
        closed_set_fp.close()
        closed_set = filter(lambda fn: fn != '', closed_set)
        closed_set = set(closed_set)
    else:
        closed_set = set()

    fp_closed_set = open(fn_rosbag_list,'a')

    for fullfn in rosbagfiles:
        fn = osp.basename(fullfn)
        #print(fn.split(".")[0])
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
            rgbs[cam_id] = topic

        for topic, cam_id, camera_type, depth_type in depth_groups:
            depth_msg = get_topic(fullfn, topic)
            depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            if depth.shape[1] < 600: # Too small image
                continue
            if depth.max() > 100.: # Convert [mm] to [m]
                depth /= 1000.
            if camera_type == "k4a": # Too poor
                continue
            rgb_msg = get_topic(fullfn, rgbs[cam_id])
            rgb = bridge.imgmsg_to_cv2(rgb_msg,desired_encoding="bgr8" )
            rgb = cv2.resize(rgb,(400,300))

            lap3 = cv2.Laplacian(depth, cv2.CV_32FC1, ksize=3)
            lap3 = cv2.resize(lap3,(1280,960))
            lap5 = cv2.Laplacian(depth, cv2.CV_32FC1, ksize=5)
            lap5 = cv2.resize(lap5,(1280,960))

            #print(topic, cam_id, camera_type, depth_type)
            if n_frames['train'] < usages['train']*len(closed_set) :
                usage = 'train'
            else:
                usage = 'valid'

            usagepath = osp.join(dataset_path,usage)
            fn_depth = osp.join(usagepath,'%d_depth.npy'%n_frames[usage])
            fn_info  = osp.join(usagepath,'%d_info.txt'%n_frames[usage])
            fp_info = open(fn_info, 'w')
            line = '\t'.join([fn, topic, cam_id, camera_type, depth_type])

            fp_info.write(line+"\n")
            fp_info.flush()

            fp_closed_set.write(fn+"\n")
            fp_closed_set.flush()

            n_frames[usage] += 1

            cv2.imshow("lap3", (lap3<-0.01).astype(np.uint8)*255)
            cv2.imshow("lap5", (lap5<-0.1 ).astype(np.uint8)*255)
            cv2.imshow("rgb", rgb)
            c = cv2.waitKey()
            if c == ord('q'):
                exit(1)


            # TODO write image
            # TODO kolourpaint iamge
            # TODO update rosbaglist.txt
            # TODO show instance map with component 



