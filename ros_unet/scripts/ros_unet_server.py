#!/usr/bin/python3
#-*- coding:utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
import cv2
import torch
from unet.unet_model import DuNet
from unet.util import SplitAdapter, ConvertDepth2input
import unet_ext as cpp_ext #TODO erase
import ros_unet.srv

import ros_numpy

class Node:
    def __init__(self, model):
        self.model = model
        self.spliter = SplitAdapter()
        self.pub_th_edge = rospy.Publisher("~th_edge", Image, queue_size=1)
        self.pub_unet_edge = rospy.Publisher("~unet_edge", Image, queue_size=1)

    def PredictEdge(self, req):
        depth = np.frombuffer(req.depth.data, dtype=np.float32).reshape(req.depth.height, req.depth.width)
        input_stack, grad, hessian, outline, convex_edge = ConvertDepth2input(depth, req.fx, req.fy)
        input_stack = torch.Tensor(input_stack).unsqueeze(0)
        input_x = self.spliter.put(input_stack).to(device)
        pred = model(input_x)
        pred = pred.detach()
        pred = self.spliter.restore(pred)
        mask = self.spliter.pred2mask(pred)

        for i in range(2):
            input_stack = np.stack( ((mask==1).astype(np.float32), grad[:,:,0], grad[:,:,1]), axis=0 )
            input_stack = torch.Tensor(input_stack).unsqueeze(0)
            input_x = self.spliter.put(input_stack).to(device)
            pred = model(input_x)
            pred = pred.detach()
            pred = self.spliter.restore(pred)
            mask = self.spliter.pred2mask(pred)

        mask_concave = np.stack((mask,convex_edge),axis=2)
        output_msg = ros_numpy.msgify(Image, mask_concave, encoding='8UC2')

        rgb = np.frombuffer(req.rgb.data, dtype=np.uint8).reshape(req.rgb.height, req.rgb.width,3)
        dst1 = (rgb.copy()/2).astype(np.uint8)
        dst1[outline==1,2] = 255
        th_edge_msg = ros_numpy.msgify(Image, dst1, encoding='8UC3')
        self.pub_th_edge.publish(th_edge_msg)

        dst2 = (rgb.copy()/2).astype(np.uint8)
        dst2[mask==1,2] = 255
        unet_edge_msg = ros_numpy.msgify(Image, dst2, encoding='8UC3')
        self.pub_unet_edge.publish(unet_edge_msg)
        return output_msg

if __name__ == '__main__':
    rospy.init_node('ros_unet_server', anonymous=True)
    fn = rospy.get_param('~weight_file')
    input_ch = rospy.get_param('~input_ch')
    device = "cuda:0"
    model = DuNet()
    model.to(device)
    checkpoint = torch.load(fn)
    model.load_state_dict(checkpoint['model_state_dict'])
    node = Node(model)

    s = rospy.Service('~PredictEdge', ros_unet.srv.ComputeEdge, node.PredictEdge)
    rospy.spin()


