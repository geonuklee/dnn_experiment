#!/usr/bin/python3
#-*- coding:utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np

if __name__ == '__main__':
    rospy.init_node('ros_unet', anonymous=True)
    rate = rospy.Rate(hz=30)

    while not rospy.is_shutdown():
        rate.sleep()
        print("done")

