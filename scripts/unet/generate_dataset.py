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


'''
TODO
0) From ground truth list
0) Ask user to overwrite exist frame, if it is already exist frame.
1) Collect all /cam*/k4a/depth/image_raw, and rgb or rgb_to_depth for it.
2) Thresholding laplacian of depth
3) Save edge - rgb. (rgb for reference)
4) Call ' kolourpaint ground_truth.png'
5) Convert above to edge.npy
6) Add 'filename' to filename list.




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
    # TODO hardcoding
    rosbag_path = '/home/geo/dataset/unloading/**/*.bag'
    files = glob2.glob(rosbag_path,recursive=True)

    bridge = CvBridge()
    closed_set = set()
    for fullfn in files:
        fn = osp.basename(fullfn,)
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

            print(topic, cam_id, camera_type, depth_type)
            cv2.imshow("lap3", (lap3<-0.01).astype(np.uint8)*255)
            cv2.imshow("lap5", (lap5<-0.1 ).astype(np.uint8)*255)
            cv2.imshow("rgb", rgb)
            c = cv2.waitKey()
            if c == ord('q'):
                exit(1)


