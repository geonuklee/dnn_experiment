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

import ros_numpy
import time

class Sub:
    def __init__(self, rgb, depth, info):
        self.sub_depth = rospy.Subscriber(depth, Image, self.cb_depth, queue_size=1)
        self.sub_rgb   = rospy.Subscriber(rgb, Image, self.cb_rgb, queue_size=1)
        self.sub_info  = rospy.Subscriber(info, CameraInfo, self.cb_info, queue_size=1)
        self.depth = None
        self.rgb = None
        self.info = None

    def cb_depth(self, topic):
        if self.depth is not None:
            return
        self.depth = ros_numpy.numpify(topic)

    def cb_rgb(self, topic):
        if self.rgb is not None:
            return
        self.rgb = ros_numpy.numpify(topic)[:,:,:3]

    def cb_info(self, topic):
        if self.info is not None:
            return
        self.info = topic

def visualize_gradient_x(grad):
    dgx = np.zeros( (grad.shape[0],grad.shape[1],3), np.uint8)
    gmax = 20.
    for r in range(grad.shape[0]):
        for c in range(grad.shape[1]):
            gx, gy = grad[r,c,:]
            if gx > 0.3:
                gx = min(gmax,gx)
                vx = 200./gmax * gx
                vx = max(vx, 0)
                dgx[r,c,2] = 255 #int(vx)
            if gx < -0.3:
                gx = min(gmax,-gx)
                vx = 200./gmax * gx
                vx = max(vx, 0)
                dgx[r,c,0] = 255 #int(vx)
    return dgx

def visualize_hessian(hessian, cv_rgb, threshold_curvature):
    dh  = cv_rgb.copy()
    dh[hessian > threshold_curvature,2] = 255
    dh[hessian > threshold_curvature,:2] = 0
    dh[hessian < -threshold_curvature,0] = 255
    dh[hessian < -threshold_curvature,1:] = 0
    return dh

def visualize_outline(hessian, dd_edg, threshold_curvature, do=None ):
    if do is None:
        do = np.zeros( (hessian.shape[0],hessian.shape[1],3), np.uint8)
    do[hessian < -threshold_curvature, 0] = 255
    do[dd_edge > 0, 1] = 255
    return do

if __name__ == '__main__':
    rospy.init_node('ros_unet', anonymous=True)
    rate = rospy.Rate(hz=100)
    cameras = rospy.get_param("~cameras")
    subs = {}
    for cam_id in cameras:
        sub = Sub("~cam%d/rgb"%cam_id, "~cam%d/depth"%cam_id, "~cam%d/info"%cam_id)
        subs[cam_id] = sub
    fn = rospy.get_param('~weight_file')
    input_ch = rospy.get_param('~input_ch')

    device = "cuda:0"
    model = DuNet()
    model.to(device)
    checkpoint = torch.load(fn)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.eval()

    spliter = SplitAdapter()
    pubs_mask, pubs_th_edge, pubs_vis_edge, pubs_vis_masks = {}, {}, {}, {}
    for cam_id in cameras:
        pub_mask = rospy.Publisher("~cam%d/mask"%cam_id, Image, queue_size=1)
        pub_th_edge = rospy.Publisher("~cam%d/th_edge"%cam_id, Image, queue_size=1)
        pub_vis_edge = rospy.Publisher("~cam%d/vis_edge"%cam_id, Image, queue_size=1)
        pubs_mask[cam_id] = pub_mask
        pubs_th_edge[cam_id] = pub_th_edge
        pubs_vis_edge[cam_id] = pub_vis_edge

    irgb, idepth, iinfo = 0, 0, 0
    while not rospy.is_shutdown():
        rate.sleep()
        for cam_id in cameras:
            sub = subs[cam_id]
            while not rospy.is_shutdown():
                if sub.depth is None:
                    rate.sleep()
                    idepth += 1
                    if idepth % (100*len(cameras)) == 0:
                        rospy.loginfo("No depth for camea %d" % cam_id)
                    continue
                else:
                    idepth = 0
                if sub.rgb is None:
                    rate.sleep()
                    irgb += 1
                    if irgb % (100*len(cameras)) == 0:
                        rospy.loginfo("No rgb for camea %d" % cam_id)
                    continue
                else:
                    irgb = 0
                if sub.info is None:
                    rate.sleep()
                    iinfo += 1
                    if iinfo % (100*len(cameras)) == 0:
                        rospy.loginfo("No info for camea %d" % cam_id)
                    continue
                else:
                    iinfo = 0
                break
            depth = sub.depth
            cv_rgb = sub.rgb
            info = sub.info
            sub.depth = None
            sub.rgb = None
            if depth is None:
                break
            if cv_rgb is None:
                break
            fx, fy = info.K[0], info.K[4]

            #dd_edge = cpp_ext.GetDiscontinuousDepthEdge(depth, threshold_depth=0.1)
            #cv2.imshow("dd_edge", 255*dd_edge.astype(np.uint8))
            #fd = cpp_ext.GetFilteredDepth(depth, dd_edge, sample_width=5)
            #grad, valid = cpp_ext.GetGradient(fd, sample_offset=3,fx=fx,fy=fy) # 5
            #hessian = cpp_ext.GetHessian(depth, grad, valid, fx=fx, fy=fy)
            #fd[ fd[:,:,0] > 2.,0 ]  = 2.
            #cv2.imshow("fdu", (fd[:,:,0]*100).astype(np.uint8))
            #threshold_curvature = 15.
            #dgx = visualize_gradient_x(grad)
            #cv2.imshow("dgx", dgx.astype(np.uint8))
            #dh = visualize_hessian(hessian, cv_rgb, threshold_curvature)
            #cv2.imshow("dh", dh)
            #outline = np.logical_or(hessian < -threshold_curvature, dd_edge > 0).astype(np.uint8)
            #cv2.imshow("outline",255*outline)
            #do = visualize_outline(hessian, dd_edge, threshold_curvature,
            #        (cv_rgb.copy()/2).astype(np.uint8) )
            #cv2.imshow("do", do)
            # cv2.waitKey(1)

            t0 = time.clock()
            # TODO ConvertDepth2input 다시 작동하게.
            # -> 왜 가끔가다 정전발생하지??
            input_stack, grad, hessian, outline = ConvertDepth2input(depth, fx, fy)

            input_stack = torch.Tensor(input_stack).unsqueeze(0)
            input_x = spliter.put(input_stack).to(device)
            pred = model(input_x)
            pred = pred.detach()
            pred = spliter.restore(pred)

            pub_mask = pubs_mask[cam_id]
            pub_th_edge, pub_vis_edge = pubs_th_edge[cam_id], pubs_vis_edge[cam_id]
            mask = spliter.pred2mask(pred)

            msg = ros_numpy.msgify(Image, mask, encoding='8UC1')
            pub_mask.publish(msg)

            cv2.imshow("outline", outline*255)
            cv2.imshow("mask", (mask==1).astype(np.uint8)*255)
            print(mask.shape)

            if pub_vis_edge.get_num_connections() > 0:
                dst = spliter.mask2dst(mask)
                dst = cv2.addWeighted(dst,0.5,cv_rgb,0.5,0)
                msg = ros_numpy.msgify(Image, dst, encoding='8UC3')
                pub_vis_edge.publish(msg)
            if pub_th_edge.get_num_connections() > 0:
                msg = ros_numpy.msgify(Image, 255*outline.astype(np.uint8), encoding='8UC1')
                pub_th_edge.publish(msg)

            print("Elapsed time = %.2f [sec]"% (time.clock()-t0) )
            #cv2.imshow("cvhessian", ( cvhessian < -100. ).astype(np.uint8)*255)
            #cv2.imshow("gx", -cvgrad[:,:,0])
            if cv2.waitKey(1) == ord('q'):
                break



