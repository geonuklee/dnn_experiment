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

from math import ceil

class SplitAdapter:
    def __init__(self, wh=128, step=100):
        self.wh, self.step = wh, step

    def put(self, x):
        assert(x.dim()==4)
        self.org_shape = x.shape
        wh, step = self.wh, self.step
        b, ch, h, w = x.shape
        dim_h, dim_w = 2, 3
    
        margin = int( (wh-step)/2 )
        nr = ceil( (h-2*margin)/step )
        nc = ceil( (w-2*margin)/step )
        padded_x = torch.zeros([b, ch, nr*step+2*margin, nc*step+2*margin], dtype=x.dtype)
        padded_x[:, :, :h, :w] = x
    
        # patches.shape = b, ch, nr, nc, w, h
        patches = padded_x.unfold(dim_h,wh,step).unfold(dim_w,wh,step)
        
        batches = torch.zeros([b*nr*nc, ch, wh,wh], dtype=patches.dtype)
        k = 0
        self.rc_indices = []
        self.b, self.nr, self.nc = b, nr, nc
        for r in range(nr):
            for c in range(nc):
                batches[b*k:b*(k+1),:,:,:] = patches[:,:,r,c,:,:]
                self.rc_indices.append((r,c))
                k+=1
        if len(self.rc_indices) == 1:
            import pdb; pdb.set_trace()
        return batches
    
    
    def restore(self, patches):
        assert(patches.dim()==4)
        wh, step, org_shape = self.wh, self.step, self.org_shape
        margin = int( (wh-step)/2 )
        nrnc_b, channels = patches.shape[:2]
        b,nr,nc = self.b, self.nr, self.nc
    
        dst = torch.zeros([b, channels, nr*step+2*margin, nc*step+2*margin],dtype=patches.dtype)
        _, _, h, w = dst.shape
        for k, (r,c) in enumerate(self.rc_indices):
            if r > 0:
                dr = margin
            else:
                dr = 0
            if c > 0:
                dc = margin
            else:
                dc = 0
            r0 = r*step+dr
            r1 = min( r0+step+margin-dr, h )
            c0 = c*step+dc
            c1 = min( c0+step+margin-dc, w)
            patch = patches[b*k:b*(k+1),:,dr:-margin, dc:-margin]
            dst[:,:,r0:r1,c0:c1] = patch
    
        dst = dst[:,:,:org_shape[-2],:org_shape[-1]]
        return dst

    def pred2mask(self, pred):
        assert(pred.dim()==4)
        assert(pred.shape[0]==1)
        pred = pred.squeeze(0).moveaxis(0,-1)
        mask = np.zeros((pred.shape[0],pred.shape[1]),np.uint8)

        edge = (pred[:,:,-1] > .9).numpy()
        mask[edge] = 1
        return mask

    def pred2dst(self, pred, np_rgb):
        mask = self.pred2mask(pred)
        dst = (np_rgb/2).astype(np.uint8)
        dst[mask>0,:2] = 0
        dst[mask>0,2] = 255
        margin = int( (self.wh-self.step)/2 )
        h, w, _ = dst.shape
        for k, (r,c) in enumerate(self.rc_indices):
            if r > 0:
                dr = margin
            else:
                dr = 0
            if c > 0:
                dc = margin
            else:
                dc = 0
            r0 = r*self.step+dr
            r1 = min( r0+self.step+margin-dr, h )
            c0 = c*self.step+dc
            c1 = min( c0+self.step+margin-dc, w)
            dst = cv2.rectangle(dst, (c0,r0), (c1,r1), (50,50,50),1)
        gray = cv2.cvtColor(np_rgb, cv2.COLOR_BGR2GRAY)
        gray = np.stack((gray,gray,gray),axis=2)
        dst = cv2.addWeighted(dst, 0.9, gray, 0.1, 0.)
        return dst

def Convert2InterInput(rgb, depth, fx, fy, threshold_curvature=20.):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    dd_edge = cpp_ext.GetDiscontinuousDepthEdge(depth, threshold_depth=0.01)
    fd = cpp_ext.GetFilteredDepth(depth, dd_edge, sample_width=5)
    grad, valid = cpp_ext.GetGradient(fd, sample_offset=0.01, fx=fx,fy=fy)
    hessian = cpp_ext.GetHessian(depth, grad, valid, fx=fx, fy=fy)

    convex_edge = (hessian > 40.).astype(np.uint8)
    concave_edge = hessian < -threshold_curvature
    outline = np.logical_or(concave_edge, dd_edge > 0).astype(np.uint8)

    hessian[hessian > threshold_curvature] = threshold_curvature
    hessian[hessian < -threshold_curvature] = -threshold_curvature


    # Normalization -.5 ~ 5
    hessian /= 2.*threshold_curvature
    # -2. ~ 2.
    hessian *= 4.

    # As a score fore outline
    hessian[dd_edge > 0] = -0.2*threshold_curvature

    max_grad = 2 # tan(60)
    grad[grad > max_grad] = max_grad
    grad[grad < -max_grad] = -max_grad
    # Normalization -.5 ~ 5
    grad /= 2.*max_grad
    input_stack = np.stack( ((gray/255.).astype(hessian.dtype),
                             hessian,
                             grad[:,:,0],
                             grad[:,:,1]
                             ), axis=0 )
    return (input_stack, grad, hessian, outline, convex_edge )

def ConvertDepth2input(depth, fx, fy):
    dd_edge = cpp_ext.GetDiscontinuousDepthEdge(depth, threshold_depth=0.1)
    #fd = np.stack((depth,depth),axis=2)
    fd = cpp_ext.GetFilteredDepth(depth, dd_edge, sample_width=5)
    grad, valid = cpp_ext.GetGradient(fd, sample_offset=0.012, fx=fx,fy=fy)
    hessian = cpp_ext.GetHessian(depth, grad, valid, fx=fx, fy=fy)

    #if np.isnan(fd.sum()):
    #    print("fd has nan")
    #    import pdb; pdb.set_trace()
    #if np.isnan(grad.sum()):
    #    print("grad has nan")
    #    import pdb; pdb.set_trace()
    #if np.isnan(hessian.sum()):
    #    print("hessian has nan")
    #    import pdb; pdb.set_trace()

    threshold_curvature = 20. # 25
    concave_edge = hessian < -threshold_curvature
    outline = np.logical_or(concave_edge, dd_edge > 0).astype(np.uint8)
    convex_edge = (hessian > 50).astype(np.uint8)

    max_grad = 2 # tan(60)
    grad[grad > max_grad] = max_grad
    grad[grad < -max_grad] = -max_grad
    # Normalization
    grad /= 2.*max_grad
    input_stack = np.stack( (outline,
                             grad[:,:,0],
                             grad[:,:,1]
                             ), axis=0 )
    return (input_stack, grad, hessian, outline, convex_edge )

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

def GetColoredLabel(marker):
    dst = np.zeros((marker.shape[0],marker.shape[1],3), dtype=np.uint8)
    uniq = np.unique(marker)
    n= len(colors)
    for u in uniq:
        if u == 0:
            continue
        dst[marker==u] = colors[u%n]
    return dst

def GetNormalizedDepth(depth):
    depth = cv2.normalize(depth, 0, 255, cv2.NORM_MINMAX)
    return depth.astype(np.uint8)

def remove_small_instance(marker, min_width=20):
    retval, marker, stats, centroids = cv2.connectedComponentsWithStats( (marker>0).astype(np.uint8) )
    outlier = set()
    for i in range(stats.shape[0]):
        if i == 0:
            continue # bg
        l,t,w,h,s = stats[i,:]
        if w < min_width:
            outlier.add(i)
        elif h < min_width:
            outlier.add(i)
    for i in outlier:
        marker[marker==i] = 0
    return marker, outlier


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


