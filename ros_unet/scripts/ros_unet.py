#!/usr/bin/python3
#-*- coding:utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
import ros_numpy
import torch

from unet.unet_model import DuNet
from unet.util import SplitAdapter
import unet_ext as cpp_ext

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
    dsize = (1280,960) # TODO remove duplicated dsize

    device = "cuda:0"
    model = DuNet()
    model.to(device)
    checkpoint = torch.load(fn)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.eval()

    spliter = SplitAdapter()

    while not rospy.is_shutdown():
        rate.sleep()
        if sub.depth is None:
            continue
        if sub.rgb is None:
            continue
        depth = sub.depth
        cv_rgb = sub.rgb
        sub.depth = None
        sub.rgb = None

        cvlap = cv2.Laplacian(depth, cv2.CV_32FC1,ksize=5)
        cvgrad = cpp_ext.GetGradient(depth, 2)

        blap = torch.Tensor(cvlap<-0.3).unsqueeze(0).unsqueeze(0).float()
        grad = torch.Tensor(cvgrad).unsqueeze(0).moveaxis(-1,1).float()
        rgb = torch.Tensor(cv_rgb).unsqueeze(0).float().moveaxis(-1,1)/255

        input_x = torch.cat((blap,grad),dim=1)
        input_x = spliter.put(input_x).to(device)
        pred = model(input_x)
        pred = pred.detach()
        pred = spliter.restore(pred)
        dst = spliter.pred2dst(pred)
        dst = cv2.addWeighted(dst,0.5,cv_rgb,0.5,0)

        ##cv2.imshow("rgb", cv_rgb)
        ##cv2.imshow("lap", cvlap)
        cv2.imshow('pred', dst)
        c = cv2.waitKey(1)
        if c == ord('q'):
            exit(1)



