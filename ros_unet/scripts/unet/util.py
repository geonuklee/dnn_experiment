#!/usr/bin/python3
#-*- coding:utf-8 -*-

import torch
import numpy as np
import unet_ext as cpp_ext
import cv2

colors = (
  (0,255,0),
  (0,180,0),
  (0,100,0),
  (255,0,255),
  (100,0,255),
  (255,0,100),
  (100,0,100),
  (0,0,255),
  (0,0,180),
  (0,0,100),
  (255,255,0),
  (100,255,0),
  (255,100,0),
  (100,100,0),
  (255,0,0),
  (180,0,0),
  (100,0,0),
  (0,255,255),
  (0,100,255),
  (0,255,100),
  (0,100,100)
)


class SplitAdapter:
    def __init__(self, w=400, offset=399):
        self.w = w
        self.offset = offset
        assert self.w >= self.offset
        self.hw_min = [] # To reconstruct output on original size.

    def restore(self, x):
        nb, c, sh, sw = x.shape
        n = len(self.hw_min)
        b = int(nb/n)
        h, w = self.origin_hw
        output=torch.zeros((b, c, h, w), dtype=x.dtype)
      
        # TODO pred 가 더 높은 값을 남기는 덮어쓰기 필요.
        k = 0
        for ib in range(b):
            for i, (hmin, wmin) in enumerate(self.hw_min):
                #print(k, hmin, wmin, '/', nb, h, w)
                if hmin+self.w > output.shape[2]:
                    output_hmax = output.shape[2]
                else:
                    output_hmax = hmin+self.w
                if wmin+self.w > output.shape[3]:
                    output_wmax = output.shape[3]
                else:
                    output_wmax = wmin+self.w
                output[ib, :, hmin:output_hmax, wmin:output_wmax] \
                        = x[k,:,:output_hmax-hmin,:output_wmax-wmin]
                k += 1

        return output

    def pred2mask(self, pred):
        mask = np.zeros((pred.shape[-2],pred.shape[-1]),np.uint8)
        if pred.shape[1] == 3:
            box_pred   = pred.moveaxis(1,3).squeeze(0)[:,:,2].numpy()
            mask[box_pred>= 0.8] = 2

        edge_pred  = pred.moveaxis(1,3).squeeze(0)[:,:,1].numpy()
        mask[edge_pred> 0.8] = 1
        return mask

    def mask2dst(self, mask):
        dst = np.zeros((mask.shape[0], mask.shape[1],3),np.uint8)
        dst[mask==1,:] = 255
        dst[mask==2, 2] = 255
        return dst

    def put(self, x):
        if x.dim() == 4:
            b, c, h0,w0 = x.shape
        else:
            b, h0,w0 = x.shape

        if len(self.hw_min) == 0:
            self.origin_hw = (h0, w0)
            for wmin in range(0, w0, self.offset):
                if wmin + self.w >= w0:
                    wmin = w0 - self.w

                if wmin < 0:
                    wmin = 0
                for hmin in range(0, h0, self.offset):
                    if hmin +self.w >= h0:
                        hmin = h0 - self.w
                    if hmin < 0:
                        hmin = 0
                    self.hw_min.append((hmin,wmin))

        if x.dim() == 4:
            output=torch.zeros((b*len(self.hw_min), c, self.w, self.w), dtype=x.dtype)
        else:
            output=torch.zeros((b*len(self.hw_min), self.w, self.w), dtype=x.dtype)

        n = 0
        if x.dim() == 4:
            for ib in range(b):
                for hmin, wmin in self.hw_min:
                    partial = x[ib,:,hmin:hmin+self.w,wmin:wmin+self.w]
                    output[n,:,:partial.shape[-2],:partial.shape[-1]] = partial 
                    n+=1
        else:
            for ib in range(b):
                for hmin, wmin in self.hw_min:
                    partial = x[ib,hmin:hmin+self.w,wmin:wmin+self.w]
                    output[n,:partial.shape[-2],:partial.shape[-1]] = partial
                    n+=1
        return output

def GetGradient(depth, fx, fy):
    cvgrad=cpp_ext.GetGradient(depth, sample_offset=5,sample_width=7,fx=fx,fy=fy)
    return cvgrad

def ConvertDepth2input(depth, fx, fy):
    cvgrad = GetGradient(depth, fx, fy)

    if False:
        cvlap=cpp_ext.GetLaplacian(depth,grad_sample_offset=1,grad_sample_width=7,fx=fx,fy=fy)
        #th_lap = -500
    else:
        cvlap=cpp_ext.GetLaplacian(depth,grad_sample_offset=5,grad_sample_width=7,fx=fx,fy=fy)
        #th_lap = -100

    max_grad = 2 # tan(60)
    cvgrad[cvgrad > max_grad] = max_grad
    cvgrad[cvgrad < -max_grad] = -max_grad

    # curvature_min define sensitivity
    curvature_min = 1. / 0.01 #  1./( radius[meter] )

    cv_bedge = ( cvlap < -curvature_min ).astype(np.uint8)
    cv_wrinkle = ( np.abs(cvlap) > curvature_min ).astype(np.uint8)

    # Normalization
    cvgrad /= 2.*max_grad

    input_stack = np.stack( (cv_bedge, #cv_wrinkle,
                             cvgrad[:,:,0],
                             cvgrad[:,:,1]
                             ), axis=0 )
    return input_stack, cvgrad, cvlap, cv_bedge, cv_wrinkle

def AddEdgeNoise(edge):
    n_noise = 200
    noise_radius = 10
    min_offset = int(noise_radius * 3)

    width, height = edge.shape[1], edge.shape[0]
    mask = np.zeros_like(edge)

    while n_noise > 0:
        candidate = np.argwhere( np.logical_and(edge==1, mask==0) )
        n_candidate = candidate.shape[0]
        if n_candidate == 0:
            break
        i = np.random.choice(range(n_candidate))
        y,x = candidate[i,:]

        cv2.circle(mask, (x,y), min_offset, 1, -1)
        cv2.circle(edge, (x,y), noise_radius, 0, -1)
        n_noise -= 1
    cv2.imshow("mask", mask*255)
    return edge

def get_meterdepth(depth):
    if depth.max() > 100.: # Convert [mm] to [m]
         return depth/ 1000.
    return depth

if __name__ == '__main__':
    from segment_dataset import CombinedDatasetLoader
    dataloader = CombinedDatasetLoader(batch_size=1)
    spliter = SplitAdapter(w=100, offset=99)
    for i, data in enumerate(dataloader):
        source = data['source']
        if source != 'labeled':
            continue
        orgb = data['rgb'].moveaxis(-1,1) #.squeeze(0).numpy().astype(np.uint8)
        rgb = spliter.put(orgb)
        drgb = spliter.restore(rgb)
        drgb = drgb.moveaxis(1,-1).squeeze(0).numpy().astype(np.uint8)
        cv2.imshow("orgb", orgb.moveaxis(1,-1).squeeze(0).numpy().astype(np.uint8) )
        cv2.imshow("drgb", drgb)
        if ord('q')==cv2.waitKey():
            exit(1)

