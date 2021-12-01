#!/usr/bin/python3
#-*- coding:utf-8 -*-

import torch

class SplitAdapter:
    def __init__(self, model=None):
        self.model = model
        self.w = 100
        self.offset = 80
        assert self.w > self.offset
        self.hw_min = [] # To reconstruct output on original size.

    def restore(self, x):
        nb, c, sh, sw = x.shape
        n = len(self.hw_min)
        b = int(nb/n)
        h, w = self.origin_hw
        output=torch.zeros((b, c, h, w), dtype=x.dtype)
       
        k = 0
        for ib in range(b):
            for i, (hmin, wmin) in enumerate(self.hw_min):
                #print(k, hmin, wmin, '/', nb, h, w)
                output[ib, :, hmin:hmin+self.w, wmin:wmin+self.w] = x[k,:,:,:]
                k += 1

        return output

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

                for hmin in range(0, h0, self.offset):
                    if hmin +self.w >= h0:
                        hmin = h0 - self.w
                    self.hw_min.append((hmin,wmin))

        if x.dim() == 4:
            output=torch.zeros((b*len(self.hw_min), c, self.w, self.w), dtype=x.dtype)
        else:
            output=torch.zeros((b*len(self.hw_min), self.w, self.w), dtype=x.dtype)

        n = 0
        if x.dim() == 4:
            for ib in range(b):
                for hmin, wmin in self.hw_min:
                    output[n,:,:,:] = x[ib,:,hmin:hmin+self.w,wmin:wmin+self.w]
                    n+=1
        else:
            for ib in range(b):
                for hmin, wmin in self.hw_min:
                    output[n,:,:] = x[ib,hmin:hmin+self.w,wmin:wmin+self.w]
                    n+=1
        return output


