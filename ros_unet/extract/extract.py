#!/usr/bin/env python
#-*- coding:utf-8 -*-
#ros_unet/extract/extract.py

import numpy as np
from scipy import misc
import cv2


if __name__ == "__main__":
    edge = cv2.imread('edge.png',cv2.IMREAD_GRAYSCALE)
    dist = cv2.distanceTransform( (~edge).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    ndist = (255 * dist/dist.max() ).astype(np.uint8)

    dst = np.ones( (edge.shape[0],edge.shape[1],3), dtype=np.uint8)*255
    for i in range(5):
        if i == 0:
            continue
        dth = 25*i
        mask = dist > dth
        _, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for idx in range(hierarchy.shape[1]):
            ar = cv2.minAreaRect(contours[idx])
            w,h = ar[1]

            #if min(w,h) < 40:
                #cv2.drawContours(dst,contours, idx, (0,0,255), 2)
            #else:
            cv2.drawContours(dst,contours, idx, (0,0,0), 2)

    dst[edge>0,0] = 150 
    dst[edge>0,1] = 150
    dst[edge>0,2] = 150

    dst = dst[2:-2,2:-2,:] 
    cv2.imshow('dst', dst)
    cv2.imwrite('cnt_example.png', dst)
    cv2.waitKey()
