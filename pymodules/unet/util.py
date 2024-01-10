#!/usr/bin/python3
#-*- coding:utf-8 -*-

import torch
import numpy as np
import unet_ext as cpp_ext
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose
import cv2

colors = (
  (0,180,0),
  (0,100,0),
  (255,0,255),
  (100,0,255),
  (100,0,100),
  (0,0,180),
  (0,0,100),
  (255,255,0),
  (100,255,0),
  (100,100,0),
  (100,0,0),
  (0,255,255),
  (0,100,255),
  (0,255,100),
  (0,100,100)
)

from math import ceil, floor

class SplitAdapter:
    def __init__(self, wh=128, step=100):
        # TODO Change input param as nr, nc
        pass

    def put(self, x):
        assert(x.dim()==4)
        self.org_shape = x.shape
        b, ch, h, w = x.shape

        nr, nc = 2,3
        self.border = 50
        m = 2*self.border

        wm = ceil( ( w+m*(nc-1) )/ nc )
        hm = ceil( ( h+m*(nr-1) )/ nr )
        
        self.tiles = []
        for v0 in range(0,h-m,hm-m):
            v1 = v0 + hm
            if v1 > h:
                dx = v1 -h
                v0 -= dx
                v1 -= dx
            for u0 in range(0,w-m,wm-m):
                u1 = u0 + wm
                if u1 > w:
                    dx = u1 -h
                    u0 -= dx
                    u1 -= dx
                #print( "%3d:%3d, %3d:%3d"%(u0,u1, v0,v1) )
                self.tiles.append( (v0,v1,u0,u1) )
        batches = torch.zeros([b*len(self.tiles), ch, hm, wm], dtype=x.dtype)
        for i, (v0,v1,u0,u1) in enumerate(self.tiles):
            batches[b*i:b*(i+1),:,:,:] = x[:,:,v0:v1,u0:u1]
        return batches
    
    def restore(self, patches):
        assert(patches.dim()==4)
        dst_shape = [self.org_shape[0], patches.shape[1], self.org_shape[2], self.org_shape[3]]
        dst = torch.zeros(dst_shape, dtype=patches.dtype)
        b, ch, h, w = dst.shape
        for i, (v0,v1,u0,u1) in enumerate(self.tiles):
            x0, y0, x1, y1 = 0, 0, patches.shape[3], patches.shape[2]
            if u0 > 0:
                u0 += self.border
                x0 += self.border
            if v0 > 0:
                v0 += self.border
                y0 += self.border
            if u1 < w:
                u1 -= self.border
                x1 -= self.border
            if v1 < h:
                v1 -= self.border
                y1 -= self.border
            # Get inner area only
            #print( "%3d:%3d, %3d:%3d"%(u0,u1, v0,v1) )
            dst[:,:,v0:v1,u0:u1] = patches[b*i:b*i+b,:,y0:y1,x0:x1]
        return dst

    def pred2mask(self, pred, th=.9):
        assert(pred.dim()==4)
        assert(pred.shape[0]==1)
        pred = pred.squeeze(0).moveaxis(0,-1)
        mask = np.zeros((pred.shape[0],pred.shape[1]),np.uint8)
        if pred.shape[-1] == 1:
            edge = (pred[:,:,-1] > th).numpy()
            mask[edge] = 1
        else:
            outline = (pred[:,:,0] > th).numpy()
            convex_edges = (pred[:,:,1] > th).numpy()
            mask[outline] = 1 # Overwrite outline edge on convex edge.
            mask[convex_edges] = 2
        return mask

    def pred2dst(self, pred, np_rgb, th=.9):
        #mask = self.pred2mask(pred, th)
        mask0 = self.pred2mask(pred, th=.5)
        mask1 = self.pred2mask(pred, th=.9)

        dst = (np_rgb/2).astype(np.uint8)
        dst[mask0==1,:] = 0
        dst[mask0==1,0] = 255
        dst[mask1==1,2] = 255

        #dst[mask==1,:2] = 0
        #dst[mask==1,2] = 255
        #dst[mask==2,1:] = 0
        #dst[mask==2,0] = 255
        #gray = cv2.cvtColor(np_rgb, cv2.COLOR_BGR2GRAY)
        #gray = np.stack((gray,gray,gray),axis=2)
        #dst = cv2.addWeighted(dst, 0.9, gray, 0.1, 0.)
        return dst

def Convert2IterInput(depth, fx, fy, rgb=None, threshold_curvature=20.):
    dd_edge = cpp_ext.GetDiscontinuousDepthEdge(depth, threshold_depth=0.02)
    fd = cpp_ext.GetFilteredDepth(depth, dd_edge, sample_width=10)
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

    if rgb is None:
        input_stack = np.stack( (hessian,
                                 grad[:,:,0],
                                 grad[:,:,1]
                                 ), axis=0 )
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
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

def GetColoredLabel(marker, text=False):
    dst = np.zeros((marker.shape[0],marker.shape[1],3), dtype=np.uint8)
    uniq = np.unique(marker)
    n= len(colors)
    for u in uniq:
        if u == 0:
            continue
        part = marker==u
        color = colors[u%n]
        dst[part] = color
        if not text:
            continue
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats( (part).astype(np.uint8) )
        color = (255-color[0], 255-color[1], 255-color[2])
        for i, (x0,y0,w,h,s) in enumerate(stats):
            if w == marker.shape[1] and h == marker.shape[0]:
                continue
            cp = centroids[i].astype(np.int)
            msg = '%d'%u
            w,h = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN,1.5,2)[0]
            cv2.rectangle(dst,(cp[0]-2,cp[1]+5),(cp[0]+w+2,cp[1]-h-5),(255,255,255),-1)
            cv2.rectangle(dst,(cp[0]-2,cp[1]+5),(cp[0]+w+2,cp[1]-h-5),(100,100,100),1)
            cv2.putText(dst, msg, (cp[0],cp[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    return dst

def Overlap(marker, rgb, text=False):
    dst = np.zeros((marker.shape[0],marker.shape[1],3), dtype=np.uint8)
    dst_rgb = rgb.copy()
    uniq = np.unique(marker)
    n= len(colors)
    for u in uniq:
        if u == 0:
            continue
        part = marker==u
        color = colors[u%n]
        dst[part] = color
        if not text:
            continue
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats( (part).astype(np.uint8) )
        color = (255-color[0], 255-color[1], 255-color[2])
        for i, (x0,y0,w,h,s) in enumerate(stats):
            if w == marker.shape[1] and h == marker.shape[0]:
                continue
            cp = centroids[i].astype(np.int)
            msg = '%d'%u
            w,h = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN,1.5,2)[0]
            for img in [dst, dst_rgb]:
                cv2.rectangle(img,(cp[0]-2,cp[1]+5),(cp[0]+w+2,cp[1]-h-5),(255,255,255),-1)
                cv2.rectangle(img,(cp[0]-2,cp[1]+5),(cp[0]+w+2,cp[1]-h-5),(100,100,100),1)
                cv2.putText(img, msg, (cp[0],cp[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2)
    dst = cv2.addWeighted(dst_rgb, .3, dst, .7, 0.)
    return dst


def Evaluate2D(obb_resp, gt_marker, rgb, strict_rule=True):
    pred_marker = np.frombuffer(obb_resp.marker.data, dtype=np.int32)\
            .reshape(obb_resp.marker.height, obb_resp.marker.width)
    pred_filtered_outline = np.frombuffer(obb_resp.filtered_outline.data, dtype=np.uint8)\
            .reshape(obb_resp.marker.height, obb_resp.marker.width)
    # ref : https://stackoverflow.com/questions/24780697/numpy-unique-list-of-colors-in-the-image
    gt_pred = np.stack((gt_marker, pred_marker), axis=2)
    pair_marker, counts = np.unique(gt_pred.reshape(-1,2),axis=0, return_counts=True)
    #cv2.imshow("gt",GetColoredLabel(gt_marker,True))
    #cv2.imshow("pred",GetColoredLabel(pred_marker,True))
    #cv2.waitKey()
    
    # Strict rule for determine over, under seg
    if strict_rule:
        underseg_th = 0. # helio_2023-03-04-14-48-40.bag 참고
    else:
        underseg_th = 1.

    arr = []
    for (gidx, pidx), count in zip(pair_marker,counts):
        if 0 in [gidx, pidx]:
            continue
        arr.append( (gidx, pidx, count) )
    arr = np.array(arr, dtype=[('gidx',int),('pidx',int),('count',float)] )

    undersegs, oversegs = set(), set()
    ious, recalls = {}, {}
    max_precisions, g2maxpidx = {},{}

    pred_indices, tmp = np.unique(pred_marker, return_counts=True)
    pred_areas ={}
    for (pidx, ps) in zip(pred_indices, tmp.astype(np.float) ):
        if pidx<1:
            continue
        pred_areas[pidx] = ps

    gt_indices, tmp = np.unique(gt_marker, return_counts=True)
    gt_maxwidth = {}
    for gidx in gt_indices:
        if gidx<1:
            continue
        gt_part = gt_marker==gidx
        gt_part[0,:] = gt_part[:,0] = gt_part[-1,:] = gt_part[:,-1] = 0
        dist = cv2.distanceTransform(gt_part.astype(np.uint8),distanceType=cv2.DIST_L2, maskSize=5)
        gt_maxwidth[gidx]=dist.max()
    gt_areas ={}
    for (gidx, gs) in zip(gt_indices, tmp.astype(np.float) ):
        if gidx<1:
            continue
        gt_areas[gidx] = gs
        matches = arr[arr['gidx']==gidx]
        matches = np.flip( matches[np.argsort(matches, order='count')] )
        if matches.shape[0] == 0:
            iou = 0.
            recall = 0.
            pidx = -1
            precision = 0.
        else:
            pidx = matches['pidx'][0]
            intersection = matches['count'][0]
            recall    = intersection/gs
            precision = intersection/pred_areas[pidx]
            iou       = intersection/float(np.logical_or(gt_marker==gidx,pred_marker==pidx).sum())
            overlap_pred = []
            for _pidx, intersection in matches[['pidx','count']]:
                if _pidx < 1:
                    continue
                if intersection/pred_areas[_pidx] > .8:
                    overlap_pred.append(_pidx)
            if len(overlap_pred) > 1:
                '''
                * 모든 pred에 대해, (pred-gt)intersection outline 안쪽 최대폭이
                  gt width의 x%를 넘을 경우 overseg라 판정.
                * 해당 gt 크기에 비해 너무 작은 pred는 무시하기 위함.
                * Overseg 평가와 관련해서, helio_2023-03-04-15-00-36.bag
                '''
                overseg = True
                if not strict_rule:
                    gt_part = gt_marker==gidx
                    for _pidx in overlap_pred:
                        intersec_part = np.logical_and(pred_marker==_pidx, gt_part)
                        intersec_part[0,:] = intersec_part[:,0] = intersec_part[-1,:] = intersec_part[:,-1] = 0
                        intersec_dist = cv2.distanceTransform( intersec_part.astype(np.uint8),
                                distanceType=cv2.DIST_L2, maskSize=5)
                        intersec_width = intersec_dist.max()
                        if intersec_width < .2 * gt_maxwidth[gidx]:
                            overseg=False
                            break
            else:
                overseg = False
                
            if overseg:
                oversegs.add(gidx)
        recalls[gidx] = recall
        ious[gidx] = iou
        g2maxpidx[gidx] = pidx
        max_precisions[gidx] = precision 
    pred_indices, pred_areas = np.unique(pred_marker, return_counts=True)
    for (pidx, ps) in zip(pred_indices, pred_areas.astype(np.float) ):
        if pidx == 0:
            continue
        matches = arr[arr['pidx']==pidx]
        matches = np.flip( matches[np.argsort(matches, order='count')] )
        if matches.shape[0] == 0:
            precision = 0.
        else:
            precision = matches['count'][0] / ps
            overlap_gt, underseged_gt = [], []
            for gidx, intersection in matches[['gidx','count']]:
                if gidx < 1:
                    continue
                if intersection/gt_areas[gidx] > .2:
                    overlap_gt.append(gidx)
            if len(overlap_gt) > 1:
                pred_part = pred_marker==pidx
                for gidx in overlap_gt:
                    gt_part = gt_marker==gidx
                    gt_part[0,:] = gt_part[:,0] = gt_part[-1,:] = gt_part[:,-1] = False
                    gt_dist = cv2.distanceTransform((~gt_part).astype(np.uint8),distanceType=cv2.DIST_L2, maskSize=5)
                    farthest  = gt_dist[pred_part].max()
                    '''
                    * pred instance의 안쪽에서, gt outline 바깥으로 거리 최대값이
                      gt width의 x%를 넘을 경우 underseg라 판정.
                    * 해당 gt에 비해 너무 작은 이웃 gt instance는 무시하기 위함.
                    * Underseg 평가와 관련해서, helio_2023-03-04-14-48-40.bag, gt#12 참고
                    '''
                    if farthest > underseg_th * gt_maxwidth[gidx]:
                        underseged_gt.append(gidx)
            for gidx in underseged_gt:
                undersegs.add(gidx)

    outputlist = []
    dst = np.zeros((gt_marker.shape[0],gt_marker.shape[1],3), dtype=np.uint8)
    #dst[gt_marker<1,:] = 0
    for gidx in gt_indices:
        if gidx == 0:
            continue
        iou, recall, precision = ious[gidx], recalls[gidx], max_precisions[gidx]
        pidx = g2maxpidx[gidx]
        overseg, underseg = gidx in oversegs, gidx in undersegs
        output = gidx, iou, recall, overseg, underseg, pidx, precision
        outputlist.append(output)
        part = gt_marker==gidx
        if overseg:
            dst[part,0] = 255
        if underseg:
            dst[part,2] = 255
        if not overseg and not underseg:
            dst[part,:] = rgb[part,:]

    boundary = cpp_ext.GetBoundary(gt_marker, 2)
    dst_rgb = rgb.copy()
    dst_rgb[boundary>0,:] = dst[boundary>0,:] = 0
    dst = cv2.addWeighted(dst_rgb, .3, dst, .7, 0.)
    dst_pred = Overlap(pred_marker,rgb,True)
    for gidx in gt_indices:
        if gidx<1:
            continue
        part = gt_marker == gidx
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats( part.astype(np.uint8) )
        for i, (x0,y0,w,h,s) in enumerate(stats):
            if w == gt_marker.shape[1] and h == gt_marker.shape[0]:
                continue
            pt = centroids[i].astype(np.int)
            msg = '%d'%gidx
            w,h = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN,1.5,2)[0]
            #cv2.rectangle(dst,(pt[0]-2,pt[1]+5),(pt[0]+w+2,pt[1]-h-5),(255,255,255),-1)
            #cv2.rectangle(dst,(pt[0]-2,pt[1]+5),(pt[0]+w+2,pt[1]-h-5),(100,100,100),1)
            cv2.putText(dst, msg, (pt[0],pt[1]), cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,0),2)
    dst = np.hstack((dst_pred,dst))

    msg = ' #g/ #p/  IoU / Recall/ '
    font_face, font_scale, font_thick = cv2.FONT_HERSHEY_PLAIN, 1., 1
    w,h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
    hoffset = 5
    w,h = w+2,h+hoffset
    dst_score = np.zeros((dst.shape[0],w,3),dst.dtype)
    cp = [0,h]
    cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, (255,255,255), font_thick)
    cp[1] += h+hoffset
    for output in outputlist:
        gidx, iou, recall, overseg, underseg, pidx, precision = output
        cp[0] = 0
        msg = '%2d'% gidx
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, (255,255,255), font_thick)
        cp[0] += w
        msg = '  %2d' % pidx
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, (255,255,255), font_thick)
        if  iou < .6:
            color = (0,0,255)
        else:
            color = (255,255,255)
        cp[0] += w
        msg = '   %.3f'%iou
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, color, font_thick)
        if  recall < .5:
            color = (0,0,255)
        else:
            color = (255,255,255)
        cp[0] += w
        msg = '   %.3f'%recall
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, color, font_thick)

        cp[1] += h+hoffset

    dst = np.hstack((dst,dst_score))
    #cv2.imshow("dst", dst)
    #if ord('q') == cv2.waitKey(len(oversegs)==0):
    #    exit(1)

    return outputlist, dst


def GetNormalizedDepth(depth):
    depth = cv2.normalize(depth, 0, 255, cv2.NORM_MINMAX)
    return depth.astype(np.uint8)

def remove_small_instance(marker0, min_width=20):
    retval, marker, stats, centroids = cv2.connectedComponentsWithStats( (marker0>0).astype(np.uint8) )
    outliers = set()
    for i in range(stats.shape[0]):
        if i == 0:
            continue # bg
        l,t,w,h,s = stats[i,:]
        if w < min_width:
            outliers.add(i)
        elif h < min_width:
            outliers.add(i)
    output = marker0.copy()
    output_outlier = set()
    for outlier in outliers:
        outlier0 = np.unique(marker0[marker==outlier])
        for o0 in outlier0:
            output_outlier.add(o0)
            output[output==o0] = 0
    return output, output_outlier


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


