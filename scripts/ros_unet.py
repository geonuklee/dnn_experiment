#!/usr/bin/python3
#-*- coding:utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
import ros_numpy
import torch

from unet.unet_model import IterNet
from unet.util import SplitAdapter

class Sub:
    def __init__(self, depth, rgb):
        self.sub_depth = rospy.Subscriber(depth, Image, self.cb_depth, queue_size=1)
        self.sub_rgb = rospy.Subscriber(rgb, Image, self.cb_rgb, queue_size=1)
        self.depth = None
        self.rgb = None

    def cb_depth(self, topic):
        if self.depth is not None:
            return
        self.depth = ros_numpy.numpify(topic)

    def cb_rgb(self, topic):
        if self.rgb is not None:
            return
        self.rgb = ros_numpy.numpify(topic)


if __name__ == '__main__':
    rospy.init_node('ros_unet', anonymous=True)
    rate = rospy.Rate(hz=10)
    sub = Sub(depth='~input_depth', rgb='~input_rgb')
    fn = rospy.get_param('~weight_file')

    device = "cuda:0"
    model = IterNet()
    model.to(device)
    state_dict = torch.load(fn)
    model.load_state_dict(state_dict)
    spliter = SplitAdapter()

    while not rospy.is_shutdown():
        rate.sleep()
        if sub.depth is None:
            continue
        if sub.rgb is None:
            continue
        depth = sub.depth
        rgb = sub.rgb
        sub.depth = None
        sub.rgb = None

        cvlap = cv2.Laplacian(depth, cv2.CV_32FC1,ksize=5)
        lap = torch.Tensor(cvlap).unsqueeze(0).unsqueeze(0).float()
        lap = spliter.put(lap).to(device)

        pred = model(lap)
        pred = pred.detach()
        pred = spliter.restore(pred)
        dst = spliter.pred2dst(pred)

        cv2.imshow("rgb", rgb)
        cv2.imshow("lap", cvlap)
        cv2.imshow('pred', dst)
        cv2.waitKey(1)


