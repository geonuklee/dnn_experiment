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
        th = .9
        if pred.shape[-1] == 1:
            edge = (pred[:,:,-1] > th).numpy()
            mask[edge] = 1
        else:
            outline = (pred[:,:,0] > th).numpy()
            convex_edges = (pred[:,:,1] > th).numpy()
            mask[outline] = 1 # Overwrite outline edge on convex edge.
            mask[convex_edges] = 2
        return mask

    def pred2dst(self, pred, np_rgb):
        mask = self.pred2mask(pred)
        dst = (np_rgb/2).astype(np.uint8)
        dst[mask==1,:2] = 0
        dst[mask==1,2] = 255
        dst[mask==2,1:] = 0
        dst[mask==2,0] = 255

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


def Evaluate2D(obb_resp, gt_marker, rgb):
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
            for pidx, intersection in matches[['pidx','count']]:
                if pidx < 1:
                    continue
                if intersection/pred_areas[pidx] > .8:
                    overlap_pred.append(pidx)
            # TODO recall대신 gidx boundary중, overlap_pred 로부터 20pixel 이상 떨어진 원소가 있는경우.
            if len(overlap_pred) > 1:
                overseg = False
                gt_mask = gt_marker==gidx
                for pidx in overlap_pred:
                    part = pred_marker==pidx
                    part[0,:] = part[:,0] = part[-1,:] = part[:,-1] = 0
                    dist_part = cv2.distanceTransform( (~part).astype(np.uint8),
                            distanceType=cv2.DIST_L2, maskSize=5)
                    farthest = np.max(dist_part[gt_mask])
                    if farthest > 20.:
                        overseg=True
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
            overlap_gt = []
            for gidx, intersection in matches[['gidx','count']]:
                if gidx < 1:
                    continue
                if intersection/gt_areas[gidx] > .2:
                    overlap_gt.append(gidx)
            # TODO precision대신 pidx boundary중, overlap_gt로부터 20pixel 이상 떨어진 원소가 있는경우.
            if len(overlap_gt) > 1:
                underseg = False
                pred_mask = pred_marker==pidx
                for gidx in overlap_gt:
                    part = gt_marker==gidx
                    part[0,:] = part[:,0] = part[-1,:] = part[:,-1] = 0
                    dist_part = cv2.distanceTransform( (~part).astype(np.uint8),
                            distanceType=cv2.DIST_L2, maskSize=5)
                    farthest = np.max(dist_part[pred_mask])
                    if farthest > 20.:
                        underseg=True
                        break
            else:
                underseg = False

            if underseg:
                for gidx in overlap_gt:
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
            cv2.rectangle(dst,(pt[0]-2,pt[1]+5),(pt[0]+w+2,pt[1]-h-5),(255,255,255),-1)
            cv2.rectangle(dst,(pt[0]-2,pt[1]+5),(pt[0]+w+2,pt[1]-h-5),(100,100,100),1)
            cv2.putText(dst, msg, (pt[0],pt[1]), cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,0),2)
    dst = np.hstack((dst_pred,dst))

    msg = ' # /  IoU / Recall/ '
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
        msg = '%2d' % gidx
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

    return outputlist, pred_marker, dst


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


