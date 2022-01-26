#!/usr/bin/python3
#-*- coding:utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
import cv2
import torch

from unet.unet_model import DuNet
from unet.util import SplitAdapter
import unet_ext as cpp_ext

import ros_numpy

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
    pubs_mask, pubs_blap, pubs_vis_edge, pubs_vis_masks = {}, {}, {}, {}
    for cam_id in cameras:
        pub_mask = rospy.Publisher("~cam%d/mask"%cam_id, Image, queue_size=1)
        pub_blap = rospy.Publisher("~cam%d/blap"%cam_id, Image, queue_size=1)
        pub_vis_edge = rospy.Publisher("~cam%d/vis_edge"%cam_id, Image, queue_size=1)
        pubs_mask[cam_id] = pub_mask
        pubs_blap[cam_id] = pub_blap
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
            cvgrad=cpp_ext.GetGradient(depth,sample_offset=5,sample_width=7,fx=fx,fy=fy)

            if False:
                cvlap=cpp_ext.GetLaplacian(depth,grad_sample_offset=1,grad_sample_width=7,fx=fx,fy=fy)
                th_lap = -500
            else:
                cvlap=cpp_ext.GetLaplacian(depth,grad_sample_offset=5,grad_sample_width=7,fx=fx,fy=fy)
                th_lap = -100
            cv_bedge = cvlap < th_lap

            max_grad = 2
            cvgrad[cvgrad > max_grad] = max_grad
            cvgrad[cvgrad < -max_grad] = -max_grad
            curvature_min = 1. / 0.01
            cvlap[cvlap > curvature_min] = curvature_min
            cvlap[cvlap < -curvature_min] = -curvature_min

            bedge = torch.Tensor(cv_bedge).unsqueeze(0).unsqueeze(0).float()
            grad = torch.Tensor(cvgrad).unsqueeze(0).moveaxis(-1,1).float()
            rgb = torch.Tensor(cv_rgb).unsqueeze(0).float().moveaxis(-1,1)/255

            if input_ch == 3: # If edge detection only
                input_x = torch.cat((bedge,grad),dim=1)
            elif input_ch == 6: # If try to detect edge and box, both of them.
                input_x = torch.cat((bedge,grad,rgb),dim=1)
            else:
                assert(False)

            input_x = spliter.put(input_x).to(device)
            pred = model(input_x)
            pred = pred.detach()
            pred = spliter.restore(pred)

            pub_mask = pubs_mask[cam_id]
            pub_blap, pub_vis_edge = pubs_blap[cam_id], pubs_vis_edge[cam_id]
            mask = spliter.pred2mask(pred)

            msg = ros_numpy.msgify(Image, mask, encoding='8UC1')
            pub_mask.publish(msg)

            if pub_vis_edge.get_num_connections() > 0:
                dst = spliter.mask2dst(mask)
                dst = cv2.addWeighted(dst,0.5,cv_rgb,0.5,0)
                msg = ros_numpy.msgify(Image, dst, encoding='8UC3')
                pub_vis_edge.publish(msg)
            if pub_blap.get_num_connections() > 0:
                msg = ros_numpy.msgify(Image, 255*cv_bedge.astype(np.uint8), encoding='8UC1')
                pub_blap.publish(msg)

            #cv2.imshow("blap", 255*(cv_bedge).astype(np.uint8) )
            cv2.imshow("cvlap", cvlap)
            cv2.imshow("gx", cvgrad[:,:,0])
            cv2.moveWindow("gx", 700, 50)
            cv2.imshow("gy", cvgrad[:,:,1])
            cv2.moveWindow("gy", 700, 600)

            if cv2.waitKey(1) == ord('q'):
                break



