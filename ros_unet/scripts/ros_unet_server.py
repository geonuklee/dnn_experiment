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

def getMask(rough_mask):
    r = 10
    dist0 = cv2.distanceTransform( (rough_mask>0).astype(np.uint8),
            distanceType=cv2.DIST_L2, maskSize=5)
    thick_parts = (dist0 > r).astype(np.uint8)
    dist1 = cv2.distanceTransform( (thick_parts<1).astype(np.uint8),
            distanceType=cv2.DIST_L2, maskSize=5)
    thin_edges = np.logical_and(dist0>0, dist1 > r+1).astype(np.uint8)

    #cv2.imshow("dist0", 0.01*dist0)
    #cv2.imshow("dist1", 0.01*dist1)
    ##cv2.imshow("rgb", rgb)
    #cv2.imshow("m0", 255*rough_mask)
    #cv2.imshow("m1", 255*thick_parts.astype(np.uint8))
    #cv2.imshow("m2", 255*thin_edges)
    #dst = 255*np.stack((thin_edges, np.zeros_like(rough_mask), rough_mask),axis=2)
    #cv2.imshow("dst", dst)
    #if ord('q') == cv2.waitKey(1):
    #    exit(1)
    return thin_edges


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
        input_x = torch.Tensor(input_x).unsqueeze(0)
        y1, y2, pred = self.model(input_x)
        del y1, y2, input_x
        pred = pred.to('cpu')
        pred = self.model.spliter.restore(pred)
         # .9 for 22-05-06-20-11-00
        mask0 = self.model.spliter.pred2mask(pred, th=.4) == 1
        mask0 = getMask(mask0.astype(np.uint8) )
        mask1 = self.model.spliter.pred2mask(pred, th=.9)
        mask = mask1.copy()
        mask[mask0>0] = 1
        del pred
        #mask[edges>0] = 0

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

            dst2 = (rgb/2).astype(np.uint8)
            dst2[mask0==1,:] = 0
            dst2[mask0==1,0] = 255

            dst2[mask1==1,:] = 0
            dst2[mask1==1,2] = 255
            #cv2.imwrite("/home/geo/ws/dnn_experiment/ros_unet/tmp.png", dst2)
            cany_msg = ros_numpy.msgify(Image, dst2, encoding='8UC3')
            self.pub_canyedges.publish(cany_msg)
        if self.pub_unet_edge.get_num_connections() > 0:
            dst2 = (rgb/2).astype(np.uint8)
            #dst2[mask0==1,:] = 0
            #dst2[mask0==1,0] = 255
            #dst2[mask1==1,2] = 255
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


