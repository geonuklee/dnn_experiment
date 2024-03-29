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

import deepdish as dd
import subprocess
from util import *


def get_topic(filename, topic):
    bag = rosbag.Bag(filename)
    messages = []
    for topic, msg, t in bag.read_messages(topics=[topic]):
        messages.append(msg)
    print("len(%s) = %d" % (topic, len(messages))  )
    return messages

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
        makedirs(osp.join(dataset_path,'src') )
        for usage in usages.keys():
            usagepath = osp.join(dataset_path,'src',usage)
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

    for fullfn in rosbagfiles:
        fn = osp.basename(fullfn)
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

            minirgb  = cv2.resize(orgb, (400,400) )

            depth0 = get_meterdepth(depth0)
            # TODO 지금 lap5를 bedge 로 변경.
            lap5 = cv2.Laplacian(depth0, cv2.CV_32FC1, ksize=5)
            blap5 = (lap5<-0.1 ).astype(np.uint8)*255

            dst = np.zeros((blap5.shape[0], blap5.shape[1]+minirgb.shape[1],3), np.uint8)
            for i in range(3):
                dst[:,:blap5.shape[1],i] = blap5
                dst[minirgb.shape[0]:,blap5.shape[1]:,i] = 100
            dst[:minirgb.shape[0],blap5.shape[1]:,:] = minirgb

            cv2.imwrite("tmp.png", dst)
            callout = subprocess.call(['kolourpaint', "tmp.png"] )
            cv_gt = cv2.imread("tmp.png")[:depth0.shape[0],:depth0.shape[1],:]
            label = np.zeros((cv_gt.shape[0],cv_gt.shape[1]), np.uint8)
            label[np.logical_and(cv_gt[:,:,2] > 200, cv_gt[:,:,1] < 50)] = 2
            dist = cv2.distanceTransform((label!=2).astype(np.uint8)*255, cv2.DIST_L2,3)

            # White == Edge
            edge_range = 10 # [pixel]
            edge = np.logical_and(cv_gt[:,:,2] > 200, cv_gt[:,:,1] > 200)
            edge = np.logical_and(edge, dist < edge_range)
            label[edge] = 1

            dst_label = np.zeros((blap5.shape[0], blap5.shape[1],3), np.uint8)
            dst_label[label==1,:] = 255
            dst_label[label==2,2] = 255
            dst = np.zeros((blap5.shape[0], blap5.shape[1]+minirgb.shape[1],3), np.uint8)
            dst[:dst_label.shape[0],:dst_label.shape[1],:] = dst_label
            dst[:minirgb.shape[0],blap5.shape[1]:,:] = minirgb


            for i, depth_msg in enumerate(depth_messages):
                depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
                rgb = bridge.imgmsg_to_cv2(rgb_messages[i%len(rgb_messages)], desired_encoding="bgr8" )

                sum_frames = n_frames['train'] + n_frames['valid']
                if n_frames['train'] < usages['train']*float(sum_frames):
                    usage = 'train'
                else:
                    usage = 'valid'

                usagepath = osp.join(dataset_path,'src',usage)
                write_format = usagepath+'/%d_%s.%s'
                fp_info = open(write_format%(n_frames[usage],"info","txt"), 'w')
                line = '\t'.join([fn, topic, cam_id, camera_type, depth_type])
                fp_info.write(line+"\n")
                fp_info.flush()
                fp_info.close()

                fn_truth = osp.join(usagepath,'%d_gt.png'%n_frames[usage])
                cv2.imwrite(fn_truth, dst)

                depth = get_meterdepth(depth)
                np.save(write_format%( n_frames[usage],"depth","npy"), depth)
                np.save(write_format%( n_frames[usage],"rgb" ,"npy"), rgb)
                dd.io.save(write_format%(n_frames[usage],"caminfo","h5"), info, compression=('blosc', 9))

                n_frames[usage] += 1
                b_results_from_fn = True
                print(n_frames)


