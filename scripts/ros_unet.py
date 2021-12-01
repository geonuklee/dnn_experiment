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
    def __init__(self, depth):
        self.sub_depth = rospy.Subscriber(depth, Image, self.cb_depth, queue_size=1)
        self.depth = None

    def cb_depth(self, topic):
        if self.depth is not None:
            return
        self.depth = ros_numpy.numpify(topic)


if __name__ == '__main__':
    rospy.init_node('ros_unet', anonymous=True)
    rate = rospy.Rate(hz=10)
    sub = Sub(depth='~input_depth')
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
        depth = sub.depth
        sub.depth = None
        cvlap = cv2.Laplacian(depth, cv2.CV_32FC1,ksize=7)
        lap = torch.Tensor(cvlap).unsqueeze(0).unsqueeze(0).float()
        lap = spliter.put(lap).to(device)

        pred = model(lap)
        pred = spliter.restore(pred)
        npred = pred.moveaxis(1,3).squeeze(0)[:,:,1].detach().numpy()
        bpred = (npred > 0.5).astype(np.uint8)*255

        print(depth.shape)
        cv2.imshow("depth", depth)
        cv2.imshow("lap", cvlap)
        cv2.imshow('pred', bpred)
        cv2.waitKey(1)


