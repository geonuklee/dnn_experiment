#!/usr/bin/python
#-*- coding:utf-8 -*-

# Geonuk Lee : c++ 에서 호출하기 좋게, create_box 정의.
import numpy as np
from scipy.spatial.transform import Rotation as rotation_util
import box, iou

def create_box(quat_xyzw, t, scale_xyz):
    quat_xyzw = np.array(quat_xyzw)
    t = np.array(t)
    scale_xyz = np.array(scale_xyz)
    rot = rotation_util.from_quat(quat_xyzw)
    Rwb = rot.as_dcm()
    scale_xyz = np.array( scale_xyz )
    w = box.Box()
    b = w.from_transformation( Rwb, t, scale_xyz)
    return b

def compute_iou(b0, b1):
    loss = iou.IoU(b0, b1)
    return loss.iou(), loss.intersection()

