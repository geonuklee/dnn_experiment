#!/usr/bin/python3
#-*- coding:utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
import cv2
import torch
from unet.iternet import *
from unet.util import SplitAdapter, Convert2IterInput
import ros_unet.srv

import ros_numpy

class Node:
    def __init__(self, model):
        self.model = model
        self.pub_canyedges = rospy.Publisher("~cany_edges", Image, queue_size=1)
        self.pub_th_edge = rospy.Publisher("~th_edge", Image, queue_size=1)
        self.pub_unet_edge = rospy.Publisher("~unet_edge", Image, queue_size=1)

    def PredictEdge(self, req):
        depth = np.frombuffer(req.depth.data, dtype=np.float32).reshape(req.depth.height, req.depth.width)
        rgb = np.frombuffer(req.rgb.data, dtype=np.uint8).reshape(req.rgb.height, req.rgb.width,3)
        #input_x, grad, hessian, outline, convex_edge = Convert2IterInput(depth,req.fx,req.fy,rgb=rgb)
        input_x, grad, hessian, outline, convex_edge = Convert2IterInput(depth,req.fx,req.fy)
                #threshold_curvature=0.005)
        # Contrast filter - removing depth noise of ToF caused by high contrast.
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        #edges = cv2.Canny(gray, 240,255)
        edges = cv2.Canny(gray, 300, 350) # Over 300 for helio_2023-03-04-14-48-40
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k5)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k5)
        input_x[0, edges>0] = 0. # Remove Hessian caused by contrast

        input_x = torch.Tensor(input_x).unsqueeze(0)
        y1, y2, pred = self.model(input_x)
        del y1, y2, input_x
        pred = pred.to('cpu')
        pred = self.model.spliter.restore(pred)
         # .9 for 22-05-06-20-11-00
        mask = self.model.spliter.pred2mask(pred, th=.9)
        del pred
        mask[edges>0] = 0

        mask[depth < .001] = 1
        mask_convex = np.stack((mask==1,mask==2),axis=2).astype(mask.dtype)
        output_msg = ros_unet.srv.ComputeEdgeResponse()
        output_msg.edge = ros_numpy.msgify(Image, mask_convex, encoding='8UC2')
        output_msg.grad = ros_numpy.msgify(Image, grad, encoding='32FC2')

        if self.pub_th_edge.get_num_connections() > 0 or self.pub_canyedges.get_num_connections()>0:
            dst1 = (rgb/2).astype(np.uint8)
            dst1[outline==1,2] = 255
            th_edge_msg = ros_numpy.msgify(Image, dst1, encoding='8UC3')
            self.pub_th_edge.publish(th_edge_msg)
            dst1[edges>0,1]=255
            cany_msg = ros_numpy.msgify(Image, dst1, encoding='8UC3')
            self.pub_canyedges.publish(cany_msg)
        if self.pub_unet_edge.get_num_connections() > 0:
            dst2 = (rgb/2).astype(np.uint8)
            dst2[mask==1,2] = 255
            dst2[mask==2,0] = 255
            unet_edge_msg = ros_numpy.msgify(Image, dst2, encoding='8UC3')
            self.pub_unet_edge.publish(unet_edge_msg)
        return output_msg

if __name__ == '__main__':
    rospy.init_node('ros_unet_server', anonymous=True)
    fn = rospy.get_param('~weight_file')
    device = "cuda:0"
    checkpoint = torch.load(fn)
    model_name = checkpoint['model_name']
    state = checkpoint['model_state_dict']
    #input_ch = rospy.get_param('~input_ch')
    input_ch = state['iternet.net1.block1_c1.main.0.weight'].shape[1]
    model = globals()[model_name](input_ch)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    node = Node(model)

    s = rospy.Service('~PredictEdge', ros_unet.srv.ComputeEdge, node.PredictEdge)
    rospy.spin()


