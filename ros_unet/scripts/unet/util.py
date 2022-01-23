#!/usr/bin/python3
#-*- coding:utf-8 -*-

import torch
import numpy as np

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
        edge_pred  = pred.moveaxis(1,3).squeeze(0)[:,:,1].numpy()
        mask[edge_pred> 0.95] = 1
        if pred.shape[1] == 3:
            box_pred   = pred.moveaxis(1,3).squeeze(0)[:,:,2].numpy()
            mask[box_pred> 0.9] = 2
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

if __name__ == '__main__':
    from segment_dataset import CombinedDatasetLoader
    import cv2
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


