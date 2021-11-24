#!/usr/bin/python2
#-*- coding:utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import numpy as np
from bonet import BoNet

class Sub:
    def __init__(self, topic):
        self.subscriber = rospy.Subscriber(topic, PointCloud2, self.callback, queue_size=1)

    def callback(self, cloud):
        dtype_list = ros_numpy.point_cloud2.fields_to_dtype(cloud.fields, cloud.point_step)
        array = np.fromstring(cloud.data, dtype_list)
        # array['rgb'] -> rgb['r'], rgb['g'], rgb['b']
        rgb = ros_numpy.point_cloud2.split_rgb_field(array)
        self.xyz = np.vstack((array['x'],array['y'],array['z'])).transpose()
        self.rgb = np.vstack((rgb['r'], rgb['g'], rgb['b'])).transpose()
        print self.xyz.shape

if __name__ == '__main__':
    rospy.init_node('ros_bonet', anonymous=True)
    sub = Sub(topic='~input')
    rate = rospy.Rate(hz=30)

    while not rospy.is_shutdown():
        rate.sleep()
        if not hasattr(sub,'cloud'):
            continue

