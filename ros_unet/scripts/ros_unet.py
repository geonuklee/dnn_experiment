#!/usr/bin/python3
#-*- coding:utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
import torch

from unet.unet_model import DuNet
from unet.util import SplitAdapter
import unet_ext as cpp_ext

import ros_numpy

class Sub:
    def __init__(self, rgb, depth):
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
        self.rgb = ros_numpy.numpify(topic)[:,:,:3]


def GetDifferential(depth, ksize, compute_size):
    org_size = (depth.shape[1],depth.shape[0])
    # TODO Parameterize
    ddepth = cv2.resize(depth, compute_size, interpolation=cv2.INTER_CUBIC)
    cvlap = cv2.Laplacian(ddepth, cv2.CV_32FC1,ksize=ksize)
    cvgrad = cpp_ext.GetGradient(ddepth, 2)
    # Restore size
    cvlap = cv2.resize(cvlap, org_size, interpolation=cv2.INTER_NEAREST)
    cvgrad = cv2.resize(cvgrad, org_size, interpolation=cv2.INTER_NEAREST)
    return cvlap, cvgrad

if __name__ == '__main__':
    rospy.init_node('ros_unet', anonymous=True)
    rate = rospy.Rate(hz=100)
    cameras = rospy.get_param("~cameras")
    subs = {}
    for cam_id in cameras:
        sub = Sub("~cam%d/rgb"%cam_id, "~cam%d/depth"%cam_id)
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

    irgb, idepth = 0, 0
    while not rospy.is_shutdown():
        rate.sleep()
        for cam_id in cameras:
            sub = subs[cam_id]
            while True:
                if sub.depth is None:
                    rate.sleep()
                    idepth += 1
                    if idepth % (100*len(cameras)) == 0:
                        print("No depth for camea %d" % cam_id)
                    continue
                else:
                    idepth = 0
                if sub.rgb is None:
                    rate.sleep()
                    irgb += 1
                    if irgb % (100*len(cameras)) == 0:
                        print("No rgb for camea %d" % cam_id)
                    continue
                else:
                    irgb = 0
                break
            depth = sub.depth
            cv_rgb = sub.rgb
            sub.depth = None
            sub.rgb = None

            #cvlap = cv2.Laplacian(depth, cv2.CV_32FC1,ksize=5)
            #cvgrad = cpp_ext.GetGradient(depth, 2)
            cvlap, cvgrad = GetDifferential(depth, ksize=5, compute_size=(1280,960))
            blap_th = -0.2


            blap = torch.Tensor(cvlap<blap_th).unsqueeze(0).unsqueeze(0).float()
            grad = torch.Tensor(cvgrad).unsqueeze(0).moveaxis(-1,1).float()
            rgb = torch.Tensor(cv_rgb).unsqueeze(0).float().moveaxis(-1,1)/255

            if input_ch == 3: # If edge detection only
                input_x = torch.cat((blap,grad),dim=1)
            elif input_ch == 6: # If try to detect edge and box, both of them.
                input_x = torch.cat((blap,grad,rgb),dim=1)
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
                blap = blap.moveaxis(0,-1).squeeze(0).numpy().astype(np.uint8)
                msg = ros_numpy.msgify(Image, 255*blap, encoding='8UC1')
                pub_blap.publish(msg)


