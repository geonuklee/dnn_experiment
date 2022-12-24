#!/usr/bin/python2
#-*- coding:utf-8 -*-

import rospy
import pickle
import glob2 # For recursive glob for python2
import re
import os
from os import path as osp
import numpy as np
import rosbag
import ros_unet.srv

import sensor_msgs, std_msgs
import geometry_msgs
import cv2
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose

from scipy.spatial.transform import Rotation as rotation_util
from collections import OrderedDict as Od

from myadjust_text import myadjust_text
from adjustText import adjust_text

from evaluator import get_pkg_dir, get_pick, GetMarkerCenters, VisualizeGt, marker2box, FitAxis, GetSurfCenterPoint
from Objectron import box, iou

from ros_client import *
from unet.gen_obblabeling import GetInitFloorMask
from os import makedirs
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.ticker import MultipleLocator
from tabulate import tabulate

import pyautogui
import shutil
from unet.util import GetColoredLabel, Evaluate2D
from unet_ext import GetBoundary

DPI = 75
FIG_SIZE = (24,10)
FIG_SUBPLOT_ADJUST = {'wspace':0.3, 'hspace':1.2} # 'top':FIG_TOP
N_FIG = (5,3)
FIG_TOP  = .95
FONT_SIZE = 10
XLABEL_COORD = {'x':1.05, 'y':-0.08}
XLABEL_COORD2 = {'x':1.1, 'y':-0.08}
LEGNED_ARGS={'fontsize':FONT_SIZE, 'bbox_to_anchor':(0.5, 1.3),'loc':'center'}

import matplotlib
# https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# https://stackoverflow.com/questions/23824687/text-does-not-work-in-a-matplotlib-label

def get_topicnames(bagfn, bag, given_camid='cam0'):
    depth = '/%s/helios2/depth/image_raw'%given_camid
    info  = '/%s/helios2/camera_info'%given_camid
    rgb   = '/%s/aligned/rgb_to_depth/image_raw'%given_camid
    return rgb, depth, info

def get_camid(fn):
    base = osp.splitext( osp.basename(fn) )[0]
    groups = re.findall("(.*)_(cam0|cam1)", base)[0]
    return groups[1]

def get_topics(bridge, pkg_dir,gt_fn, pick, set_cameras,floordetector_set_camera, compute_floor):
    rosbag_fn = osp.join(pkg_dir, pick['rosbag_fn'] )
    bag = rosbag.Bag(rosbag_fn)
    rgb_topics, depth_topics, info_topics = {},{},{}
    rect_info_msgs = {}
    remap_maps = {}
    cameras = [get_camid(gt_fn)] # For each file test.
    for cam_id in cameras:
        rgb_topics[cam_id], depth_topics[cam_id], info_topics[cam_id] \
                = get_topicnames(rosbag_fn, bag, given_camid=cam_id)
        try:
            _, rgb_msg, _ = bag.read_messages(topics=[rgb_topics[cam_id]]).next()
            _, depth_msg, _ = bag.read_messages(topics=[depth_topics[cam_id]]).next()
            _, info_msg, _= bag.read_messages(topics=[info_topics[cam_id]]).next()
        except:
            continue
        rect_info_msgs[cam_id], mx, my = get_rectification(info_msg)
        remap_maps[cam_id] = (mx, my)
        for set_camera in set_cameras:
            set_camera(std_msgs.msg.String(cam_id), rect_info_msgs[cam_id])
        floordetector_set_camera(std_msgs.msg.String(cam_id), rect_info_msgs[cam_id])
    rgb_msgs, depth_msgs  = {}, {}
    topic2cam = {}
    for k,v in rgb_topics.items():
        rgb_msgs[k] = None
        topic2cam[v] = k
    for k,v in depth_topics.items():
        depth_msgs[k] = None
        topic2cam[v] = k
    set_depth = set(depth_topics.values())
    set_rgb = set(rgb_topics.values())
    fx, fy = rect_info_msgs[cam_id].K[0], rect_info_msgs[cam_id].K[4]

    rect_rgb_msg, rect_depth_msg, rect_depth, rect_rgb = rectify(rgb_msg, depth_msg, mx, my, bridge)
    cvgt_fn = osp.join(pkg_dir,pick['cvgt_fn'])
    cv_gt = cv2.imread(cvgt_fn)
    max_z = 5.
    init_floormask = GetInitFloorMask(cv_gt)
    if init_floormask is None:
        plane_c = (0., 0., 0., 99.)
        floor = np.zeros((rect_depth_msg.height,rect_depth_msg.width),np.uint8)
    else:
        floor_msg = compute_floor(rect_depth_msg, rect_rgb_msg, init_floormask)
        plane_c  = floor_msg.plane
        floor_mask = floor_msg.mask
        floor = np.frombuffer(floor_mask.data, dtype=np.uint8).reshape(floor_mask.height, floor_mask.width)
    Twc = get_Twc(cam_id)

    bag = rosbag.Bag(rosbag_fn)
    return bag, set_depth, set_rgb, topic2cam, rgb_topics, depth_topics, rgb_msgs, depth_msgs,\
            rect_info_msgs, mx, my, fx, fy, Twc, plane_c, floor

def visualize_scene(pick, eval_scene):
    gt_indices = np.unique(eval_scene['gidx'])
    dst = np.zeros_like(pick['rgb'])
    gt_marker = pick['marker']
    min_iou = .6
    msg = ' # /AP>%.1f/  Over / Under '%min_iou
    font_face, font_scale, font_thick = cv2.FONT_HERSHEY_PLAIN, 1., 1
    w,h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
    hoffset = 10
    w,h = w+2,h+hoffset
    dst_score = np.zeros((dst.shape[0],w,3),dst.dtype)
    cp = [0,h]
    cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, (255,255,255), font_thick)
    cp[1] += h+hoffset
    for gidx in gt_indices:
        data = eval_scene[eval_scene['gidx']==gidx]
        n = len(data)
        n_underseg = data['underseg'].sum()
        n_overseg = data['overseg'].sum()
        n_ap = (data['iou']> min_iou).sum()
        prob_underseg = float(n_underseg) / float(n)
        prob_overseg  = float(n_overseg) / float(n)
        ap = float(n_ap) / float(n)
        part = gt_marker == gidx
        if n_overseg > 0:
            dst[part,0] = 255
        if n_underseg > 0:
            dst[part,-1] = 255
        #print(gidx, n_overseg, n_underseg)
        if not prob_overseg and not prob_underseg:
            dst[part,:] = pick['rgb'][part,:]
        #msg = '%2d / %.3f / %.3f / %.3f' %(gidx, ap, prob_overseg, prob_underseg)
        cp[0] = 0
        msg = '%2d' % gidx
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, (255,255,255), font_thick)

        if ap < 1.:
            color = (0,0,255)
        else:
            color = (255,255,255)
        cp[0] += w
        msg = '   %.3f'%ap
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, color, font_thick)

        if n_overseg > 0:
            color = (0,0,255)
        else:
            color = (255,255,255)
        cp[0] += w
        msg = '   %.3f'%prob_overseg
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, color, font_thick)

        if n_underseg > 0:
            color = (0,0,255)
        else:
            color = (255,255,255)
        cp[0] += w
        msg = '   %.3f'%prob_underseg
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(cp), font_face, font_scale, color, font_thick)
        cp[1] += h+hoffset
        #cp = (cp[0], cp[1]+h)
        #cv2.putText(dst_score, msg, cp, font_face, font_scale, (255,255,255), font_thick)
        #h += 10

    boundary = GetBoundary(gt_marker, 2)
    dst_rgb = pick['rgb'].copy()
    dst_rgb[boundary>0,:] = dst[boundary>0,:] = 0
    dst = cv2.addWeighted(dst_rgb, .3, dst, .7, 0.)

    for gidx in gt_indices:
        part = gt_marker == gidx
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats( part.astype(np.uint8) )
        for i, (x0,y0,w,h,s) in enumerate(stats):
            if w == gt_marker.shape[1] and h == gt_marker.shape[0]:
                continue
            pt = centroids[i].astype(np.int)
            msg = '%d'%gidx
            #w,h = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN,1.5,2)
            w,h = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN,1.5,2)[0]
            cv2.rectangle(dst,(pt[0]-2,pt[1]+5),(pt[0]+w+2,pt[1]-h-5),(255,255,255),-1)
            cv2.rectangle(dst,(pt[0]-2,pt[1]+5),(pt[0]+w+2,pt[1]-h-5),(100,100,100),1)
            cv2.putText(dst, msg, (pt[0],pt[1]), cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,0),2)

    dst = np.hstack((dst,dst_score))
    return dst

def Evaluate3D(pick, gt_obb_markers, obb_resp, output2dlist):
    gt_obbs, pred_obbs = {}, {}
    for gt_marker in gt_obb_markers.markers:
        gt_obbs[gt_marker.id] = gt_marker
    for pred_marker in obb_resp.output.markers:
        pred_obbs[pred_marker.id] = pred_marker

    gt_marker, rgb = pick['marker'], pick['rgb']
    dst = np.zeros((gt_marker.shape[0],gt_marker.shape[1],3), dtype=np.uint8)
    gt_indices, tmp = np.unique(gt_marker, return_counts=True)
    boundary = GetBoundary(gt_marker, 2)
    centers = GetMarkerCenters(gt_marker)

    msg = '#g/ #p/IoU2D/Recall2D/  t_err /  deg_e / s_err '
    font_face, font_scale, font_thick = cv2.FONT_HERSHEY_PLAIN, 1., 1
    w,h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
    hoffset = 5
    w,h = w+2,h+hoffset
    dst_score = np.zeros((dst.shape[0],w,3),dst.dtype)
    pt = [0,h]
    cv2.putText(dst_score, msg, tuple(pt), font_face, font_scale, (255,255,255), font_thick)
    pt[1] += h+hoffset

    output23dlist = []
    for output in output2dlist:
        gidx, iou, recall, overseg, underseg, pidx, precision = output
        #print(gidx, pidx)
        gt_obb = gt_obbs[gidx]
        # Visualize
        part = gt_marker == gidx
        cp = centers[gidx]
        if overseg:
            dst[part,0] = 255
        if underseg:
            dst[part,2] = 255
        if not overseg and not underseg:
            dst[part,:] = rgb[part,:]

        msg = '%d'%gidx
        w,h = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN,1.5,2)[0]
        cv2.rectangle(dst,(cp[0]-2,cp[1]+5),(cp[0]+w+2,cp[1]-h-5),(255,255,255),-1)
        cv2.rectangle(dst,(cp[0]-2,cp[1]+5),(cp[0]+w+2,cp[1]-h-5),(100,100,100),1)
        cv2.putText(dst, msg, (cp[0],cp[1]), cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,0),2)

        pt[0] = 0
        msg = '%2d' % gidx
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(pt), font_face, font_scale, (255,255,255), font_thick)
        pt[0] += w
        msg = ' %3d' % pidx
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(pt), font_face, font_scale, (255,255,255), font_thick)
        pt[0] += w
        if pidx in pred_obbs:
            pred_obb = pred_obbs[pidx]
            b0 = marker2box(gt_obb)
            b1 = marker2box(pred_obb)
            b1 = FitAxis(b0, b1)
            rwb0 = rotation_util.from_dcm(b0.rotation)
            rwb1 = rotation_util.from_dcm(b1.rotation)
            daxis = 0
            cp_surf0, _ = GetSurfCenterPoint(gt_obb,daxis)
            cp_surf1, _ = GetSurfCenterPoint(pred_obb,daxis)
            dr = rwb1.inv()* rwb0
            deg_err = np.rad2deg( np.linalg.norm(dr.as_rotvec()) )

            # surf_cp is twp for cente 'p'ointr of front plane on 'w'orld frame.
            t_err = cp_surf1 - cp_surf0
            t_err = np.linalg.norm(t_err)

            # Denote that x-axis is assigned to normal of front plane.
            s_err = (b1.scale-b0.scale)[1:]
            s_err = np.abs(s_err)
            max_wh_err = max( s_err ) # [meter]
        else:
            msg = '   Failed to compute OBB'
            cv2.putText(dst_score, msg, tuple(pt), font_face, font_scale, (0,0,255), font_thick)
            pt[1] += h+hoffset
            output23dlist.append( output+(False, 999., 999., 999.) )
            continue

        if  iou < .6:
            color = (0,0,255)
        else:
            color = (255,255,255)
        msg = '   %.3f'%iou
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(pt), font_face, font_scale, color, font_thick)
        if  recall < .5:
            color = (0,0,255)
        else:
            color = (255,255,255)
        pt[0] += w
        msg = '   %.3f'%recall
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(pt), font_face, font_scale, color, font_thick)

        if  t_err > .05:
            color = (0,0,255)
        else:
            color = (255,255,255)
        pt[0] += w
        msg = '   %.3f'%t_err
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(pt), font_face, font_scale, color, font_thick)
        if  deg_err > 5.:
            color = (0,0,255)
        else:
            color = (255,255,255)
        pt[0] += w
        msg = '   %.3f'%deg_err
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(pt), font_face, font_scale, color, font_thick)
        if  max_wh_err > .05:
            color = (0,0,255)
        else:
            color = (255,255,255)
        pt[0] += w
        msg = '   %.3f'%max_wh_err
        w, h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        cv2.putText(dst_score, msg, tuple(pt), font_face, font_scale, color, font_thick)

        pt[1] += h+hoffset
        output23dlist.append( output+(True, t_err, deg_err, max_wh_err) )

    dst_rgb = rgb.copy()
    dst_rgb[boundary>0,:] = dst[boundary>0,:] = 0
    dst = cv2.addWeighted(dst_rgb, .3, dst, .7, 0.)
    dst = np.hstack((dst,dst_score))
    return output23dlist, dst

def GetMinLength(eval_data, picks):
    length_array = np.repeat(-1.,eval_data.shape)
    for base, pick in picks.items():
        obbs = {}
        for each in pick['obbs']:
            obbs[each['id']] = each
        gt_marker = pick['marker']
        for gidx in np.unique(gt_marker):
            if gidx == 0:
                continue
            indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
            scale =  obbs[gidx]['scale']
            min_length = min(scale)
            length_array[indicies] = min_length
    return length_array

def GetOblique(eval_data, picks):
    oblique_array = np.repeat(-1.,eval_data.shape)
    depthvec = np.array((0.,0.,1.))
    for base, pick in picks.items():
        obbs = {}
        for each in pick['obbs']:
            obbs[each['id']] = each
        gt_marker = pick['marker']
        w,h = float(gt_marker.shape[1]), float(gt_marker.shape[0])
        for gidx in np.unique(gt_marker):
            if gidx == 0:
                continue
            indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
            t_cb = obbs[gidx]['pose'][:3]
            qw,qx,qy,qz = obbs[gidx]['pose'][3:] # w,x,y,z
            rot_cb = rotation_util.from_quat([qx, qy, qz, qw])
            nvec = rot_cb.as_dcm()[:,0]
            deg = np.rad2deg(np.arccos(-nvec.dot(depthvec)))
            oblique_array[indicies] = deg
    assert( (oblique_array<0).sum() == 0)
    return oblique_array

def GetDistance(eval_data, picks):
    distance_array = np.repeat(-1.,eval_data.shape)
    for base, pick in picks.items():
        obbs = {}
        for each in pick['obbs']:
            obbs[each['id']] = each
        gt_marker = pick['marker']
        w,h = float(gt_marker.shape[1]), float(gt_marker.shape[0])
        for gidx in np.unique(gt_marker):
            if gidx == 0:
                continue
            indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
            t_cb = obbs[gidx]['pose'][:3]
            distance_array[indicies] = np.linalg.norm(t_cb)
            #distance_array[indicies] = t_cb[2]
    assert( (distance_array<0).sum() == 0)
    return distance_array 

def GetMargin(eval_data, picks):
    margin_array = np.repeat(-1,eval_data.shape)
    minwidth_array = np.repeat(-1,eval_data.shape)
    for base, pick in picks.items():
        gt_marker = pick['marker']
        w,h = float(gt_marker.shape[1]), float(gt_marker.shape[0])
        K = pick['K']
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        for gidx in np.unique(gt_marker):
            if gidx == 0:
                continue
            indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
            part = gt_marker==gidx
            part[0,:] = part[:,0] = part[-1,:] = part[:,-1] = 0
            dist_part = cv2.distanceTransform( part.astype(np.uint8),
                    distanceType=cv2.DIST_L2, maskSize=5)
            loc = np.unravel_index( np.argmax(dist_part,axis=None), gt_marker.shape)
            dy = float(min(loc[0], h-loc[0]))
            dx = float(min(loc[1], w-loc[1]))
            #dx /= fx
            #dy /= fy
            d = min(dx, dy)
            margin_array[indicies] = d
            minwidth_array[indicies] = dist_part[loc]
    assert( (margin_array<0).sum() == 0)
    return margin_array, minwidth_array

def GetTags(eval_data, picks, tags):
    tag_array = np.repeat('',eval_data.shape).astype(object)
    for base, pick in picks.items():
        gt_marker = pick['marker']
        for gidx in np.unique(gt_marker):
            if gidx == 0:
                continue
            indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
            key = (base,gidx)
            if key in tags:
                tag_array[indicies] = tags[key]
    return tag_array

    #valid = np.repeat(True,eval_data.shape)
    #for base, pick in picks.items():
    #    gt_marker = pick['marker']
    #    for gidx in np.unique(gt_marker):
    #        if gidx == 0:
    #            continue
    #        indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
    #        if (base,gidx) in tags: # tag = tags[(base,gidx)]
    #            valid[indicies] = False
    #return valid

def LabelHeight(ax, rects, form='%.2f'):
    fig = ax.get_figure()
    texts = []
    xs, ys = [],[]
    for rect in rects:
        value = rect.get_height()
        y = value
        if y == 0.:
            continue
        x = rect.get_x()+.5*rect.get_width()
        va = 'bottom'
        txt = ax.text(x, y, form%value,
                fontsize=FONT_SIZE, ha='center', va=va) #, bbox=dict(boxstyle='square,pad=.3'))
        xs.append(x)
        ys.append(y)
        texts.append(txt)
    ren = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=ren)
    for txt in texts:
        tbbox = txt.get_window_extent(renderer=ren)
        if tbbox.ymax - bbox.ymax > -10.:
            txt.set_verticalalignment('top')
    #myadjust_text(texts,
    #        x = xs, y = ys,
    #        ax = ax,
    #        autoalign=False,
    #        precision = .001,
    #        ha = 'center',
    #        va = 'center',
    #        on_basemap=False,
    #        only_move={'text':'xy','points':'y'},
    #        text_from_points=False,
    #        force_text = (.4, .4), force_points=(.4,.4),
    #        expand_text=(2.,2.),
    #        lim=1000)
    return

def PlotLengthOblique(picks, mvbb_data, eval_data):
    datas = Od()
    datas['TODO(Naming)'] = eval_data
    datas['mvbb'] = mvbb_data

    min_length = .1

    n_bins = 6
    step = 10.
    x = np.arange(n_bins)
    min_max = [0., n_bins*step]
    max_bound = 0.
    nbar = 2
    width = 1. / float(nbar) - .05
    x = np.arange(n_bins)
    offset = float(nbar-1)*width/2.

    fig = plt.figure(figsize=(6,4), dpi=100)
    ax = fig.add_subplot(1,1,1)
    for i, (method, data) in enumerate(datas.items()):
        lengths = GetMinLength(data, picks)
        values = data[lengths > min_length]['deg_err']
        if method=='mvbb':
            print("~~~~~~~~~~~~~~~~~~~~~")
            print("%s median(deg_err) = %f"%(method,np.median(values)) )
            print("~~~~~~~~~~~~~~~~~~~~~")
        tp_hist,  bins = np.histogram(values, n_bins, min_max)
        tp_hist = tp_hist.astype(float) / np.sum(tp_hist).astype(float)
        nbar = 1
        rects = ax.bar(x-offset, width=width, height=tp_hist, alpha=.5,label=method)
        LabelHeight(ax,rects)
        offset -=width
        for bound, y in zip(np.flip(bins[1:]), np.flip(tp_hist)):
            if y > 0.:
                break
            max_bound = max(bound/step,max_bound)
    ax.set_xlim(-width, max_bound)
    ax.set_xlabel('[deg]', fontsize=FONT_SIZE)
    ax.set_ylabel('Probability', fontsize=FONT_SIZE)
    ax.legend(loc='upper right', fontsize=FONT_SIZE)
    xlabels = []
    for i in range(n_bins):
        #msg = '%.1f~%.1f'%(bins[i],bins[i+1])
        msg = '~%.1f'%(bins[i+1])
        xlabels.append(msg)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=FONT_SIZE)
    fig.tight_layout()
    return fig

def Plot2dEval(eval_data, picks, margin, valid, ax, min_iou, num_bins, min_max,
        unit_str, _format,
        show_underseg=False, show_overseg=False, show_sample=False):
    la = np.logical_and
    n_hist , bound    = np.histogram(margin[valid], num_bins,min_max)
    tp_hist , _       = np.histogram(margin[
        logical_ands([eval_data['iou']>min_iou,
                      valid,
                      ~eval_data['overseg'],
                      ~eval_data['underseg'] ])], num_bins,min_max)
    underseg_hist , _ = np.histogram(margin[la(eval_data['underseg'],valid)], num_bins,min_max)
    overseg_hist , _  = np.histogram(margin[la(eval_data['overseg'],valid)], num_bins,min_max)
    no_samples = n_hist==0
    n_hist[no_samples] = 1 # To prevent divide by zero
    nbar= 1
    if show_underseg:
        if underseg_hist.sum()==0:
            show_underseg = False
        else:
            nbar+=1
    if show_overseg:
        if overseg_hist.sum()==0:
            show_overseg = False
        else:
            nbar+=1
    tp_hist = tp_hist.astype(float) / n_hist.astype(float)
    underseg_hist = underseg_hist.astype(float) / n_hist.astype(float)
    overseg_hist  = overseg_hist.astype(float) / n_hist.astype(float)
    n_hist[no_samples] = 0
    width = 1. / float(nbar) - .05
    x = np.arange(num_bins)
    offset = float(nbar-1)*width/2.
    ap_label = 'AP(IoU >%.1f)'%min_iou
    rects = ax.bar(x-offset, width=width, height=tp_hist, alpha=.5, label=ap_label)
    LabelHeight(ax, rects)
    offset -= width
    ncol = 1
    if show_underseg:
        rects = ax.bar(x-offset, width=width, height=underseg_hist, alpha=.5, label='$p(\mathrm{under})$')
        LabelHeight(ax, rects)
        offset -= width
        ncol += 1
    if show_overseg:
        rects = ax.bar(x-offset, width=width, height=overseg_hist, alpha=.5, label='$p(\mathrm{over})$')
        LabelHeight(ax, rects)
        offset -= width
        ncol += 1
    xlabels = []
    for i in range(num_bins):
        #msg = '%.f~%.f'%(bound[i],bound[i+1])
        msg = _format%(bound[i],bound[i+1])
        if show_sample:
            msg += '\n%d'%n_hist[i]
        xlabels.append(msg)
    ax.set_xlabel('%s'%unit_str,rotation=0, fontsize=FONT_SIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    ax.xaxis.set_label_coords(**XLABEL_COORD)
    ax.yaxis.set_label_coords(-0.08, 1.)

    if nbar > 1:
        ax.legend(ncol=ncol,**LEGNED_ARGS)
    else:
        ax.set_ylabel(ap_label,rotation=0, fontsize=FONT_SIZE, fontweight='bold')

    #import pdb; pdb.set_trace()
    #fig = ax.get_figure()
    #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #ext2 = ax.xaxis.label.get_window_extent()
    return


def PlotEachScens(eval_data, picks, eval_dir, infotype='false_detection'):
    min_iou = .6
    bbox=dict(boxstyle="square", ec=(0., 1., 0., 0.), fc=(1., 1., 1.,.8) )

    for scene_idx, (base, pick) in enumerate(picks.items()):
        fig = plt.figure(figsize=(10,6), dpi=100)
        plt.subplots_adjust(wspace=0, hspace=0)
        ax = fig.add_subplot(111)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        plt.axis('off')
        valid = eval_data['base']==base
        scene_data = eval_data[valid]
        rgb, gt_marker = pick['rgb'], pick['marker']
        dst = np.zeros_like(rgb)
        gt_indices = np.unique(gt_marker)
        headers = ('AP', '','')
        scores = {}
        for gidx in gt_indices:
            if gidx==0:
                continue
            data = scene_data[scene_data['gidx']==gidx]
            n = len(data)
            n_underseg = data['underseg'].sum()
            n_overseg = data['overseg'].sum()
            n_ap = (data['iou']> min_iou).sum()
            prob_underseg = float(n_underseg) / float(n)
            prob_overseg  = float(n_overseg) / float(n)
            ap = float(n_ap) / float(n)
            scores[gidx] = (ap, prob_overseg, prob_underseg)
            part = gt_marker == gidx
            if infotype=='':
                dst[part,:] = rgb[part,:]
            else:
                if n_overseg > 0:
                    dst[part,0] = 255
                if n_underseg > 0:
                    dst[part,0] = 255
                    #dst[part,-1] = 255
                if not prob_overseg and not prob_underseg:
                    if ap < 1.:
                        dst[part,0] = 255
                        #dst[part,1] = 255
                    else:
                        dst[part,:] = rgb[part,:]

        boundary = GetBoundary(gt_marker, 2)
        dst_rgb = rgb.copy()
        dst_rgb[boundary>0,:] = dst[boundary>0,:] = 0
        dst = cv2.addWeighted(dst_rgb, .3, dst, .7, 0.)
        height, width = dst.shape[:2]
        height, width = float(height), float(width)
        ax.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), extent = [0,1,0,1],
                    aspect=height/width )
        r = 0.
        ax.set_xlim(-r,1.+r)
        ax.set_ylim(-r,1.+r)
        centers = GetMarkerCenters(gt_marker)
        objs = []
        texts = []
        for gidx, cp in centers.items():
            (ap, prob_overseg, prob_underseg) = scores[gidx]
            msg = ''
            if infotype == 'false_detection':
                if ap < 1. :
                    msg += '\nAP: %.2f'%ap
                if prob_overseg > 0. :
                    msg += '\np(Over): %.2f'%prob_overseg
                if prob_underseg > 0.:
                    msg += '\np(Under): %.2f'%prob_underseg
                if len(msg) == 0:
                    continue
                msg = msg[1:]
            elif infotype=='id':
                msg = '#%d'% gidx
            if len(msg) > 0:
                txt = ax.text(float(cp[0])/width, 1.-float(cp[1])/height, msg, fontsize=15, bbox=bbox,
                        ha='left', va='center')
                texts.append(txt)
        if infotype == 'false_detection':
            myadjust_text(texts,
                    only_move={'points':'xy', 'text':'xy', 'objects':'xy'},
                    ha='center',va='center',
                    autoalign=True,
                    precision = .001,
                    expand_text = (1.4,1.4),
                    expand_points = (1.4,1.4),
                    force_text = (.4, .4), force_points=(.4,.4),
                    text_from_text=True,
                    text_from_points=False,
                    on_basemap = False,
                    arrowprops=dict(arrowstyle="->", color=(1.,1.,0.), lw=2.),
                    lim=1000
                    )
        fig.savefig(osp.join(eval_dir,'scene%d.svg'%(scene_idx+1) ),
                bbox_inches='tight', transparent=True, pad_inches=0)
    return

def Plot3dEval(eval_data, ax, ytype,
        xdata, valid, unit_str, show_sample=True,
        num_bins=3, min_max=(.5, 3.) ):
    '''
    * 'Bar' Median error
    * 상자 물리적크기,이미지크기, 2D IoU, 이미지 위치. ->  t_err, deg_err, s_err
    * ref : https://matplotlib.org/stable/gallery/axes_grid1/parasite_simple.html#sphx-glr-gallery-axes-grid1-parasite-simple-py
    '''
    la = np.logical_and
    ax_deg = ax.twinx()
    ax.set_ylabel('[cm]', fontsize=FONT_SIZE)
    ax_deg.set_ylabel('[deg]', fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    ax_deg.tick_params(axis='y', labelsize=FONT_SIZE)
    n_hist , bound    = np.histogram(xdata[valid], num_bins,min_max)
    medians = {}
    stddevs = {}
    maes = {}
    for k in ['trans_err', 'max_wh_err', 'deg_err']:
        medians[k] = []
        stddevs[k] = []
        maes[k] = []

    for i in range(len(bound)-1):
        vmin, vmax = bound[i:i+2]
        vdiff = vmax-vmin
        in_bound = valid
        in_bound = la( xdata>vmin, in_bound)
        in_bound = la( xdata<=vmax, in_bound)
        for k in medians.keys():
            data = eval_data[k][in_bound]
            if len(data) == 0:
                median=0.
                stddev=0.
                mae = 0.
            else:
                median = np.median(data)
                stddev = np.std(data)
                mae = np.sum(np.abs(data)) / float(len(data))
            if k != 'deg_err':
                median *= 100.
                stddev *= 100.
                mae    *= 100.
            medians[k].append(median)
            stddevs[k].append(stddev)
            maes[k].append(mae)

    nbar= 3
    width = 1. / float(nbar) - .02
    offset = float(nbar-1)*width/2.
    x = np.arange(num_bins).astype(float)
    #xlim = ax.get_xlim()
    ax.set_xlim(x[0]-2.*offset,x[-1]+2.*offset)

    # https://stackoverflow.com/questions/11774822/matplotlib-histogram-with-errorbars
    #yvalues = rmses
    if ytype.lower() == 'median':
        yvalues = medians
    elif ytype.lower() == 'mae':
        yvalues = maes
    rects = ax.bar(x-offset, width=width, height=yvalues['trans_err'],
            alpha=.5, label='trans error')
    LabelHeight(ax, rects, form='%.2f')
    offset -= width

    rects = ax.bar(x-offset, width=width, height=yvalues['max_wh_err'],
            alpha=.5, label='size error')
    LabelHeight(ax, rects, form='%.2f')
    offset -= width

    rects = ax_deg.bar(x-offset, width=width, height=yvalues['deg_err'],
            alpha=.5, label='rotation error', color='green')
    LabelHeight(ax_deg, rects, form='%.1f')
    offset -= width

    xlabels = []
    for i in range(num_bins):
        msg = '%.1f~%.1f'%(bound[i],bound[i+1])
        if show_sample:
            msg += '\n%d'%n_hist[i]
        xlabels.append(msg)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=FONT_SIZE)
    ax.set_xlabel('%s'%unit_str,rotation=0, fontsize=FONT_SIZE, fontweight='bold')
    ax.xaxis.set_label_coords(**XLABEL_COORD2)

    miny0,maxy0 = ax.get_ylim()
    miny1,maxy1 = ax_deg.get_ylim()
    ax.set_ylim(0, max(maxy0,maxy1) )
    ax_deg.set_ylim(0, max(maxy0,maxy1) )

    legend_args={'fontsize':FONT_SIZE,
            'bbox_to_anchor':(0., 1.3),
            'loc':'center left'}
    ax.legend(ncol=2,**legend_args)
    legend_args={'fontsize':FONT_SIZE,
            'bbox_to_anchor':(1., 1.3),
            'loc':'center right'}
    ax_deg.legend(**legend_args)

    return

def PlotTagAp(eval_data, tags, ax, min_iou, show_underseg=False, show_overseg=False):
    la = np.logical_and
    def f_have(arr, word):
        return word in arr.split(';')
    have = np.vectorize(f_have)
    #valid = have(tags,'tape')
    nbar= 1
    if show_underseg:
        nbar+=1
    if show_overseg:
        nbar+=1
    width = 1. / float(nbar) - .1
    offset = float(nbar-1)*width/2.

    params = Od()
    params['AP (IoU >%.1f)'%min_iou] = logical_ands([eval_data['iou']>min_iou,
                                                     ~eval_data['overseg'],
                                                     ~eval_data['underseg']])
    if show_underseg:
        params['underseg'] = eval_data['underseg']
    if show_overseg:
        params['overseg'] = eval_data['overseg']
    
    for name, inliers in params.items():
        case_nhist = Od()
        valid = tags==''
        nsample = valid.sum()
        case_nhist['No specifics\n%d'%nsample]\
                = float( la(inliers, valid).sum() ) / float(nsample)

        valid = have(tags,'tape')
        nsample = valid.sum()
        case_nhist['With tape\n%d'%nsample]\
                = float( la(inliers, valid).sum() ) / float(nsample)
        x = np.array(range(len(case_nhist))).astype(float) - offset
        rects = ax.bar(x,width=width,height=case_nhist.values(), alpha=.5, label=name)
        LabelHeight(ax, rects)
        offset -= width

    ax.set_xlabel('Case',rotation=0, fontsize=FONT_SIZE, fontweight='bold')
    ax.set_xticks(range(len(case_nhist)))
    ax.set_xticklabels(case_nhist.keys(), rotation=0.,fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    #plt.tight_layout(rect=(.1, 0.,.95,.95)) # left,bottom,right,top
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    ax.legend(ncol=3,**LEGNED_ARGS)

    #ax.set_ylabel('[-]'%min_iou,rotation=0, fontweight='bold')
    ax.set_ylim(0.,1.)
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.yaxis.set_label_coords(-0.08, 1.05)
    return

def GetErrorOfMvbb(gt_obb_markers, mvbb_resp):
    gt_obbs, pred_obbs = {}, {}
    for gt_marker in gt_obb_markers.markers:
        gt_obbs[gt_marker.id] = gt_marker

    output23dlist = []
    for pred_obb in mvbb_resp.output.markers:
        gt_obb = gt_obbs[pred_obb.id]
        b0 = marker2box(gt_obb)
        b1 = marker2box(pred_obb)
        b1 = FitAxis(b0, b1)
        rwb0 = rotation_util.from_dcm(b0.rotation)
        rwb1 = rotation_util.from_dcm(b1.rotation)
        daxis = 0
        cp_surf0, _ = GetSurfCenterPoint(gt_obb,daxis)
        cp_surf1, _ = GetSurfCenterPoint(pred_obb,daxis)
        dr = rwb1.inv()* rwb0
        deg_err = np.rad2deg( np.linalg.norm(dr.as_rotvec()) )
        # surf_cp is twp for cente 'p'ointr of front plane on 'w'orld frame.
        t_err = cp_surf1 - cp_surf0
        t_err = np.linalg.norm(t_err)
        # Denote that x-axis is assigned to normal of front plane.
        s_err = (b1.scale-b0.scale)[1:]
        s_err = np.abs(s_err)
        max_wh_err = max( s_err ) # [meter]

        # Assume recall 1 because get from ground truth marker
        pidx = gidx = gt_obb.id
        iou = precision = recall = 1. 
        overseg = underseg = False
        output = gidx, iou, recall, overseg, underseg, pidx, precision 
        output23dlist.append( output+(True, t_err, deg_err, max_wh_err) )
    return output23dlist

def perform_test(eval_dir, gt_files,fn_evaldata, methods=['myobb']):
    #if osp.exists(screenshot_dir):
    #    shutil.rmtree(screenshot_dir)
    #makedirs(screenshot_dir)
    rospy.wait_for_service('~PredictEdge')
    predict_edge = rospy.ServiceProxy('~PredictEdge', ros_unet.srv.ComputeEdge)
    rospy.wait_for_service('~SetCamera')
    set_camera = rospy.ServiceProxy('~SetCamera', ros_unet.srv.SetCamera)
    rospy.wait_for_service('~ComputeObb')
    compute_obb = rospy.ServiceProxy('~ComputeObb', ros_unet.srv.ComputeObb)
    bridge = CvBridge()
    rospy.wait_for_service('~FloorDetector/SetCamera')
    floordetector_set_camera = rospy.ServiceProxy('~FloorDetector/SetCamera', ros_unet.srv.SetCamera)
    rospy.wait_for_service('~FloorDetector/ComputeFloor')
    compute_floor = rospy.ServiceProxy('~FloorDetector/ComputeFloor', ros_unet.srv.ComputeFloor)

    rospy.wait_for_service('~Cgal/ComputeCgalObb')
    compute_cgalobb = rospy.ServiceProxy('~Cgal/ComputeCgalObb', ros_unet.srv.ComputeCgalObb)
    rospy.wait_for_service('~Cgal/SetCamera')
    cgal_set_camera = rospy.ServiceProxy('~Cgal/SetCamera', ros_unet.srv.SetCamera)

    pub_gt_obb = rospy.Publisher("~gt_obb", MarkerArray, queue_size=-1)
    pub_gt_pose = rospy.Publisher("~gt_pose", PoseArray, queue_size=1)

    pkg_dir = get_pkg_dir()

    dtype = [('base',object),
            ('sidx',int), # Scene index
            ('fidx',int), # Frame index
            ('method',object), # 'myobb, mvbb, etc'
            ('gidx',int), # Ground truth object index
            ('iou',float), ('recall',float), ('overseg',bool),('underseg',bool),
            ('pidx',int), # Prediction object index
            ('precision',float),
            ('valid_obb',bool),
            ('trans_err',float),
            ('deg_err', float),
            ('max_wh_err', float),
            ]

    eval_data = None
    for i_file, gt_fn in enumerate(gt_files):
        #print(gt_fn)
        pick = get_pick(gt_fn)
        base = osp.splitext(osp.basename(pick['rosbag_fn']))[0]
        bag, set_depth, set_rgb, topic2cam, rgb_topics, depth_topics, rgb_msgs, depth_msgs,\
                rect_info_msgs, mx, my, fx, fy, Twc, plane_c, floor = \
                get_topics(bridge,pkg_dir,gt_fn, pick,
                        [set_camera, cgal_set_camera],
                        floordetector_set_camera, compute_floor)

        gt_obbs = {}
        for obb in pick['obbs']:
            gt_obbs[obb['id']] = obb
        gt_obb_poses, gt_obb_markers = VisualizeGt(gt_obbs)
        a = Marker()
        a.action = Marker.DELETEALL
        for arr in [gt_obb_markers]: # ,gt_infos
            arr.markers.append(a)
            arr.markers.reverse()

        eval_scene, nframe = [], 0
        for topic, msg, t in bag.read_messages(topics=rgb_topics.values()+depth_topics.values()):
            cam_id = topic2cam[topic]
            if topic in set_depth:
                depth_msgs[cam_id] = msg
            elif topic in set_rgb:
                rgb_msgs[cam_id] = msg

            rgb_msg, depth_msg = rgb_msgs[cam_id], depth_msgs[cam_id]
            if depth_msg is None or rgb_msg is None:
                continue
            rect_rgb_msg, rect_depth_msg, rect_depth, rect_rgb = rectify(rgb_msg, depth_msg, mx, my, bridge)
            #rect_depth[floor>0] = 0.
            #rect_depth_msg = bridge.cv2_to_imgmsg(rect_depth,encoding='32FC1')

            t0 = time.time()
            edge_resp = predict_edge(rect_rgb_msg,rect_depth_msg, fx, fy)
            plane_w = convert_plane(Twc, plane_c) # empty plane = no floor filter.
            obb_resp = compute_obb(rect_depth_msg, rect_rgb_msg, edge_resp.edge,
                    Twc, std_msgs.msg.String(cam_id), fx, fy, plane_w)
            t1 = time.time()

            if 'mvbb' in methods:
                '''
                # Comaprison for cgal obb
                * [x] collecting cases
                * [x] Cgal OBB marker array로 획득.
                * [x] deg_err 계산해서 반영.
                * [ ] 2면이 보이는 instance만 따로 골라내기 
                    * Cgal OBB둘다 일정 크기 이상의 깊이를 가지면 orientation 문제가 감지됨.
                        -> marker에 상자 두께가 관찰되는 상황이라 이야기하자.
                * [ ] deg_err의 분포 (median?)만 이야기해도 되겠다.
                '''
                marker = pick['marker']
                dist = cv2.distanceTransform( (~pick['outline']).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
                marker[dist < 5.] = 0
                marker = bridge.cv2_to_imgmsg(marker,encoding='passthrough')
                mvbb_resp = compute_cgalobb(rect_depth_msg, marker, Twc, std_msgs.msg.String(cam_id), fx, fy)
                eval_23d = GetErrorOfMvbb(gt_obb_markers, mvbb_resp)
                for each in eval_23d:
                    eval_scene.append( (base,i_file,nframe,'mvbb')+ each)

            if 'myobb' in methods:
                eval_2d, dst = Evaluate2D(obb_resp, pick['marker'], rect_rgb)
                eval_23d, dst3d = Evaluate3D(pick, gt_obb_markers, obb_resp, eval_2d)
                for each in eval_23d:
                    eval_scene.append( (base,i_file,nframe,'myobb')+ each)
            pub_gt_obb.publish(gt_obb_markers)

            cv2.imshow("frame", dst)
            cv2.imshow("dst3d", dst3d)
            if ord('q') == cv2.waitKey(1):
                exit(1)
            fn = osp.join(eval_dir, 'frame_%d_%s_%04d.png'%(i_file,base,nframe) )
            cv2.imwrite(fn, dst)
            nframe += 1
            depth_msg, rgb_msg = None, None
            if nframe >= 20:
                break

        eval_scene = np.array(eval_scene, dtype)
        if eval_data is None:
            eval_data = eval_scene
        else:
            eval_data = np.hstack((eval_data,eval_scene) )
        dst = visualize_scene(pick,eval_scene)
        cv2.imshow("scene%d"%i_file, dst)
        if ord('q') == cv2.waitKey(1):
            exit(1)
        fn = osp.join(eval_dir, 'scene_%d_%s.png'%(i_file,base) )
        cv2.imwrite(fn, dst)

    cv2.destroyAllWindows()
    with open(fn_evaldata,'wb') as f: 
        np.save(f, eval_data)

    return eval_data

def full_extent(ax, pad=0.0):
    """
        Get the full extent of an axes, including axes labels, tick labels, and
        titles.
    https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
     -> https://stackoverflow.com/a/26432947
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = []
    items += ax.get_xticklabels()
    #items += ax.get_yticklabels()
    #items.append(ax)
    items.append(ax.xaxis.label)
    items.append(ax.yaxis.label)
    items.append(ax.get_legend())
    fig = ax.get_figure()
    bbox = Bbox.union([item.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) \
            for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)

def logical_ands(list_of_arrays):
    valid = list_of_arrays[0]
    for i, arr in enumerate(list_of_arrays):
        if i == 0:
            continue
        valid = np.logical_and(valid, arr)
    return valid

def test_evaluation(show_sample):
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_test0523')
    if not osp.exists(eval_dir):
        makedirs(eval_dir)
    fn_evaldata = osp.join(eval_dir,'eval_data.npy')
    usages = ['test0523']
    gt_files = []
    for usage in usages:
        obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
        gt_files += glob2.glob(obbdatasetpath)
    if not osp.exists(fn_evaldata):
        eval_data_allmethod = perform_test(eval_dir, gt_files, fn_evaldata, methods=['myobb','mvbb'])
    else:
        with open(fn_evaldata,'rb') as f: 
            eval_data_allmethod = np.load(f, allow_pickle=True)
    picks = {}
    for fn in gt_files:
        base = osp.splitext( osp.basename(fn) )[0]
        groups = re.findall("(.*)_(cam0|cam1)", base)[0]
        base = groups[0]
        with open(fn,'r') as f:
            picks[base] = pickle.load(f)
    tags = {}
    tags['eval_test0523/scene_3_helios_test_2022-05-23-15-37-08.png']\
        = { 8:'tape'}
    tags['eval_test0523/scene_4_helios_test_2022-05-23-15-37-32.png']\
        = { 8:'tape'}
    tags['eval_test0523/scene_5_helios_2022-05-06-20-11-00.png'] = {11:'tape'}
    tags['eval_test0523/scene_6_helios_test_2022-05-23-15-52-03.png'] = {3:'tape'}
    tags['eval_test0523/scene_7_helios_scene2_2022-05-23-15-57-05.png']\
            = {6:'flat box', 9:'tape'}
    tmp_tags = {}
    for fn, val in tags.items():
        fn = osp.splitext( osp.basename(fn) )[0]
        base = re.findall("scene_\d*_(.*)", fn)[0]
        for gidx, tag in val.items():
            tmp_tags[(base,gidx)] = tag
    tags = tmp_tags
    
    la = np.logical_and
    mvbb_data = eval_data_allmethod[eval_data_allmethod['method']=='mvbb']
    eval_data = eval_data_allmethod[eval_data_allmethod['method']=='myobb']
    
    # TODO mvbb_data, eval_data 둘 비교
    fig = PlotLengthOblique(picks, mvbb_data, eval_data)
    fig.savefig(osp.join(eval_dir,'test_mvbb_oblique.svg'),
                    bbox_inches='tight', transparent=True, pad_inches=0)

    oblique = GetOblique(eval_data, picks)
    margin, minwidth = GetMargin(eval_data, picks)
    distance = GetDistance(eval_data, picks)
    tags   = GetTags(eval_data, picks, tags)
    show = True

    if True:
        axes = {}
        fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
        n = 0
        for j in range(N_FIG[1]):
            for i in range(N_FIG[0]):
                k = i*N_FIG[1]+j+1
                axes[n] = fig.add_subplot(N_FIG[0],N_FIG[1],k)
                if n < 5:
                    axes[n].yaxis.set_major_locator(MultipleLocator(.5))
                n+=1

        fig.subplots_adjust(**FIG_SUBPLOT_ADJUST)
        valid = tags==''
        vdistance = distance < 2.
        vminwidth = minwidth > 10.
        vmargin = margin > 20.
        min_iou = .6
        Plot2dEval(eval_data, picks, margin, logical_ands([valid,vminwidth,vdistance]),
                axes[0], num_bins=5, min_max=(0., 100.), unit_str='', _format='%.f~%.f',
                min_iou=min_iou, show_underseg=True, show_overseg=False, show_sample=show_sample)
        Plot2dEval(eval_data, picks, minwidth,  logical_ands([valid,vmargin,vdistance]),
                axes[1], num_bins=5, min_max=(10., 60.), unit_str='', _format='%.f~%.f',
                min_iou=min_iou, show_underseg=True, show_overseg=False, show_sample=show_sample)
        Plot2dEval(eval_data, picks, oblique, logical_ands([valid,vminwidth,vmargin,vdistance]),
                axes[2], num_bins=5, min_max=(0., 50.), unit_str='[deg]', _format='%.f~%.f',
                min_iou=min_iou, show_underseg=True, show_overseg=False, show_sample=show_sample)
        Plot2dEval(eval_data, picks, distance, logical_ands([valid,vminwidth,vmargin]),
                axes[3], num_bins=4, min_max=(.5, 3.), unit_str='[m]', _format='%.2f~%.2f',
                min_iou=min_iou, show_underseg=True, show_overseg=False, show_sample=show_sample)
        axes[0].set_title('Margin-AP',fontsize=7).set_position( (.5, 1.42))
        axes[1].set_title('min(w,h)-AP',fontsize=7).set_position( (.5, 1.42))
        axes[2].set_title('Oblique-AP',fontsize=7).set_position( (.5, 1.42))
        axes[3].set_title('Distance-AP',fontsize=7).set_position( (.5, 1.42))
        if not show_sample:
            fig.savefig(osp.join(eval_dir,'test_margin_ap.svg'), bbox_inches=full_extent(axes[0]))
            fig.savefig(osp.join(eval_dir,'test_minwidth_ap.svg'), bbox_inches=full_extent(axes[1]))
            fig.savefig(osp.join(eval_dir,'test_oblique_ap.svg'), bbox_inches=full_extent(axes[2]))
            fig.savefig(osp.join(eval_dir,'test_distance_ap.svg'), bbox_inches=full_extent(axes[3]))

        min_iou=.6
        valid = tags==''
        valid = la(valid, eval_data['iou']>min_iou)
        valid = la(valid, ~eval_data['underseg'])
        valid = la(valid, ~eval_data['overseg'])
        for i, ytype in enumerate(['Median', 'MAE']):
            n0 = 5*i
            Plot3dEval(eval_data, axes[n0+5], ytype,
                    margin, logical_ands([valid,vminwidth,vdistance]), show_sample=show_sample,
                    unit_str ='[pixel]', num_bins=5, min_max=(0., 100.) )
            # 의미있는 경향은 안보임.
            Plot3dEval(eval_data, axes[n0+6], ytype,
                    minwidth, logical_ands([valid,vmargin,vdistance]), show_sample=show_sample,
                    unit_str ='[pixel]', num_bins=7, min_max=(10, 70.) )
            Plot3dEval(eval_data, axes[n0+7], ytype,
                    oblique, logical_ands([valid,vmargin,vminwidth,vdistance]), show_sample=show_sample,
                    unit_str ='[deg]', num_bins=5, min_max=(0, 50.) )
            Plot3dEval(eval_data, axes[n0+8], ytype,
                    distance, logical_ands([valid,vmargin,vminwidth]), show_sample=show_sample,
                    unit_str ='[m]', num_bins=3, min_max=(.5, 3.) )
            axes[n0+5].set_title('Margin - Err %s'%ytype,fontsize=7).set_position( (.5, 1.42))
            axes[n0+6].set_title('min(w,h) - Err %s'%ytype,fontsize=7).set_position( (.5, 1.42))
            axes[n0+7].set_title('Oblique - Err %s'%ytype,fontsize=7).set_position( (.5, 1.42))
            axes[n0+8].set_title('Distance - Err %s'%ytype,fontsize=7).set_position( (.5, 1.42))


        validobb_data = eval_data[logical_ands([valid,vmargin,vminwidth,vdistance,eval_data['valid_obb']])]
        N = valid.sum()
        n = logical_ands([valid,eval_data['valid_obb']]).sum()
        r = float(n)/float(N)
        print("p(OBB | valid 2D seg) = %d / %d = %.f"%(n,N,r) )

        #table_data = [ ['Median', 'MAE'] ]
        table_data = [ ['trans_err', 'max_wh_err', 'deg_err'] ]
        for eval_type in ['Median', 'MAE']:
            values = [eval_type]
            for err_type in table_data[0]:
                data = validobb_data[err_type]
                if eval_type == 'Median':
                    val = np.median(data)
                else:
                    val = np.sum(np.abs(data)) / float(len(data))
                values.append(val)
            table_data.append( values )

        # ref : https://pyhdust.readthedocs.io/tabulate.html
        table = tabulate(table_data, headers="firstrow")
        print(table)
        table = tabulate(table_data, headers="firstrow", tablefmt="latex",
                floatfmt=(None,'.3f', '.3f', '.2f') )
        print(table)

        #print("3D perofrmance error

    #PlotEachScens(eval_data, picks, eval_dir, infotype='false_detection')
    return

def dist_evaluation():
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_dist')
    if not osp.exists(eval_dir):
        makedirs(eval_dir)
    fn_evaldata = osp.join(eval_dir,'eval_data.npy')
    usages = ['aligneddist']
    gt_files = []
    for usage in usages:
        obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
        gt_files += glob2.glob(obbdatasetpath)
    if not osp.exists(fn_evaldata):
        eval_data = perform_test(eval_dir, gt_files, fn_evaldata)
    else:
        with open(fn_evaldata,'rb') as f: 
            eval_data = np.load(f, allow_pickle=True)
    picks = {}
    for fn in gt_files:
        base = osp.splitext( osp.basename(fn) )[0]
        groups = re.findall("(.*)_(cam0|cam1)", base)[0]
        base = groups[0]
        with open(fn,'r') as f:
            picks[base] = pickle.load(f)
    tags = {}
    distance = GetDistance(eval_data, picks)
    margin, minwidth = GetMargin(eval_data, picks)
    valid    = margin>40.
    fig = plt.figure(1, figsize=FIG_SIZE, dpi=DPI)
    fig.subplots_adjust(**FIG_SUBPLOT_ADJUST)
    ax = fig.add_subplot(N_FIG[0],N_FIG[1],1)
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    #PlotDistanceAp(eval_data, picks, distance, valid, ax, min_iou=.5,
    #        num_bins=5,min_max=(1.,2.5), show_underseg=True, show_overseg=True)
    Plot2dEval(eval_data, picks, distance,  valid, ax,
            num_bins=5, min_max=(1., 2.5), unit_str='[m]', _format='%.2f~%.2f',
            min_iou=.5, show_underseg=True,show_overseg=True)
    fig.savefig(osp.join(eval_dir,'test_dist_ap.svg'), bbox_inches=full_extent(ax))
    PlotEachScens(eval_data, picks, eval_dir, infotype='')
    return

def oblique_evaluation():
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_oblique')
    if not osp.exists(eval_dir):
        makedirs(eval_dir)
    fn_evaldata = osp.join(eval_dir,'eval_data.npy')
    usages = ['alignedyaw', 'alignedroll']
    gt_files = []
    for usage in usages:
        obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
        gt_files += glob2.glob(obbdatasetpath)
    if not osp.exists(fn_evaldata):
        eval_data = perform_test(eval_dir, gt_files, fn_evaldata)
    else:
        with open(fn_evaldata,'rb') as f: 
            eval_data = np.load(f, allow_pickle=True)
    picks = {}
    for fn in gt_files:
        base = osp.splitext( osp.basename(fn) )[0]
        groups = re.findall("(.*)_(cam0|cam1)", base)[0]
        base = groups[0]
        with open(fn,'r') as f:
            picks[base] = pickle.load(f)
    tags = {}
    margin, minwidth = GetMargin(eval_data, picks)
    oblique = GetOblique(eval_data, picks)
    valid   = margin>40.
    fig = plt.figure(1, figsize=FIG_SIZE, dpi=DPI)
    fig.subplots_adjust(**FIG_SUBPLOT_ADJUST)
    ax = fig.add_subplot(N_FIG[0],N_FIG[1],1)
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    Plot2dEval(eval_data, picks, oblique,  valid, ax,
            num_bins=5, min_max=(0., 50.), unit_str='[deg]', _format='%.f~%.f',
            min_iou=.5, show_underseg=True,show_overseg=True)

    fig.savefig(osp.join(eval_dir,'test_oblique_ap.svg'), bbox_inches=full_extent(ax))
    PlotEachScens(eval_data, picks, eval_dir, infotype='')
    return

if __name__=="__main__":
    rospy.init_node('ros_eval', anonymous=True)
    target = rospy.get_param('~target')
    show = int(rospy.get_param('~show'))
    if target=='test':
        test_evaluation(show_sample=show)
    elif target=='dist':
        dist_evaluation()
    elif target=='oblique':
        oblique_evaluation()

    if show:
        plt.show(block=True)
