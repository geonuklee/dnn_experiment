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

            #dg  = cv_rgb.copy()
            #dg[depth[:,:]==0.,:2] = 0
            #dg[depth[:,:]==0.,2] = 255
            #dg[depth[:,:]>0.,1] += 100
            #cv2.imshow("valid_depth", dg)

            fd = cpp_ext.GetFilteredDepth(depth, sample_width=7)
            grad, valid = cpp_ext.GetGradient(fd, sample_offset=5,fx=fx,fy=fy)
            hessian = cpp_ext.GetHessian(depth, grad, valid, fx=fx, fy=fy)

            fd[ fd[:,:,0] > 2.,0 ]  = 2.
            cv2.imshow("fdu", (fd[:,:,0]*100).astype(np.uint8))
            #cv2.imshow("gx", -grad[:,:,0])
            #cv2.imshow("valid", 255*valid.astype(np.uint8))

            dgx  = cv_rgb.copy()
            dgx[grad[:,:,0]>0.,2] = 255
            dgx[grad[:,:,0]>0.,:2] = 0
            dgx[grad[:,:,0]<0.,0] = 255
            dgx[grad[:,:,0]<0.,1:] = 0
            dgx[grad[:,:,0]==0.,1] = 255
            cv2.imshow("dgx", dgx)

            dgy  = cv_rgb.copy()
            dgy[grad[:,:,1]>0.,2] = 255
            dgy[grad[:,:,1]>0.,:2] = 0
            dgy[grad[:,:,1]<0.,0] = 255
            dgy[grad[:,:,1]<0.,1:] = 0
            dgy[grad[:,:,1]==0.,1] = 255
            cv2.imshow("dgy", dgy)

            dh  = cv_rgb.copy()
            curvature = 30.;
            dh[hessian > curvature,2] = 255
            dh[hessian > curvature,:2] = 0
            dh[hessian < -curvature,0] = 255
            dh[hessian < -curvature,1:] = 0
            #dh[hessian ==0.,1] = 255
            cv2.imshow("dh", dh)

            cv2.imshow("concave", 255*(hessian < -curvature).astype(np.uint8))
            cv2.waitKey(1)
            continue # TODO Erase it after test!

            t0 = time.clock()
            input_stack, cvgrad, cvhessian, cv_bedge, cv_wrinkle = ConvertDepth2input(depth, fx, fy)
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

            if pub_vis_edge.get_num_connections() > 0:
                dst = spliter.mask2dst(mask)
                dst = cv2.addWeighted(dst,0.5,cv_rgb,0.5,0)
                msg = ros_numpy.msgify(Image, dst, encoding='8UC3')
                pub_vis_edge.publish(msg)
            if pub_th_edge.get_num_connections() > 0:
                msg = ros_numpy.msgify(Image, 255*cv_bedge.astype(np.uint8), encoding='8UC1')
                pub_th_edge.publish(msg)

            print("Elapsed time = %.2f [sec]"% (time.clock()-t0) )
            #valid_depth = (depth > 0.01).astype(np.uint8) * 200
            #cv2.imshow("valid", valid_depth)
            #print("Elapsed time = %.2f [sec]"% (time.clock()-t0) )
            #cv2.imshow("cv_wrinkle", ( cv_wrinkle > 0 ).astype(np.uint8)*255)
            cv2.imshow("cvhessian", ( cvhessian < -100. ).astype(np.uint8)*255)
            cv2.imshow("gx", -cvgrad[:,:,0])
            #cv2.imshow("mgx", (cvgrad[:,:,0]<0.).astype(np.uint8)*255)
            dgx  = cv_rgb.copy()
            dgx[cvgrad[:,:,0]>0.,2] = 255
            dgx[cvgrad[:,:,0]>0.,:2] = 0
            dgx[cvgrad[:,:,0]<0.,0] = 255
            dgx[cvgrad[:,:,0]<0.,1:] = 0
            dgx[cvgrad[:,:,0]==0.,1] = 255

            cv2.imshow("dgx", dgx)

            #cv2.moveWindow("gx", 700, 50)
            #cv2.imshow("gy", cvgrad[:,:,1])
            #cv2.moveWindow("gy", 700, 600)
            if cv2.waitKey(1) == ord('q'):
                break



