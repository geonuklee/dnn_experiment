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
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pc2

#from tf2_msgs.msg import TFMessage
import tf2_ros
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as rotation_util
from collections import OrderedDict as Od

from myadjust_text import myadjust_text
from adjustText import adjust_text

from evaluator import get_pkg_dir, get_pick, GetMarkerCenters, VisualizeGt, marker2box, FitAxis, GetSurfCenterPoint, get_topicnames, GetNeighbors
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
from unet_ext import GetBoundary, UnprojectPointscloud

DPI = 75
FIG_SIZE = (24,10)
FIG_SUBPLOT_ADJUST = {'wspace':0.3, 'hspace':1.2} # 'top':FIG_TOP
N_FIG = (5,3)
FIG_TOP  = .95
FONT_SIZE = 10
FONT_WEIGHT = None # 'bold', None
XLABEL_COORD = {'x':1.05, 'y':-0.08}
XLABEL_COORD2 = {'x':1., 'y':-0.08}
LEGNED_ARGS={'fontsize':FONT_SIZE, 'bbox_to_anchor':(0.5, 1.3),'loc':'center'}

import matplotlib
# https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# https://stackoverflow.com/questions/23824687/text-does-not-work-in-a-matplotlib-label

def get_camid(fn):
    base = osp.splitext( osp.basename(fn) )[0]
    groups = re.findall("(.*)_(cam0|cam1)", base)[0]
    return groups[1]

def get_topics(bridge, pkg_dir,gt_fn, pick):
    rosbag_fn = osp.join(pkg_dir, pick['rosbag_fn'] )
    bag = rosbag.Bag(rosbag_fn)
    rgb_topics, depth_topics, info_topics = {},{},{}
    imu_topics = {}
    info_msgs = {}
    rect_info_msgs = {}
    cameras = [get_camid(gt_fn)] # For each file test.
    for cam_id in cameras:
        rgb_topics[cam_id], depth_topics[cam_id], info_topics[cam_id], imu_topics[cam_id] \
                = get_topicnames(rosbag_fn, bag, given_camid=cam_id)
        _, rgb_msg, _ = bag.read_messages(topics=[rgb_topics[cam_id]]).next()
        _, depth_msg, _ = bag.read_messages(topics=[depth_topics[cam_id]]).next()
        _, info_msg, _= bag.read_messages(topics=[info_topics[cam_id]]).next()

        if len(info_msg.D) == 0:
            rect_info_msg = info_msg
            mx, my = None,None
        else:
            rect_info_msg, mx, my = get_rectification(info_msg)
        rect_info_msgs[cam_id] = rect_info_msg
        info_msgs[cam_id] = info_msg
    rgb_msgs, depth_msgs, imu_msgs  = {}, {}, {}
    topic2cam = {}
    for k,v in rgb_topics.items():
        rgb_msgs[k] = None
        topic2cam[v] = k
    for k,v in depth_topics.items():
        depth_msgs[k] = None
        topic2cam[v] = k
    for k,v in imu_topics.items():
        imu_msgs[k] = None
        topic2cam[v] = k

    set_depth = set(depth_topics.values())
    set_rgb = set(rgb_topics.values())
    set_imu = set(imu_topics.values())

    fx, fy = rect_info_msg.K[0], rect_info_msg.K[4]

    bag = rosbag.Bag(rosbag_fn)
    return bag, set_depth, set_rgb, set_imu, topic2cam, rgb_topics, depth_topics, imu_topics, rgb_msgs, depth_msgs, imu_msgs,\
            info_msgs, rect_info_msgs, mx,my, fx, fy#, Twc, plane_c, floor

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
            output23dlist.append( output+(False, 999., 999., 999.,None,None) )
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
        output23dlist.append( output+(True, t_err, deg_err, max_wh_err, b0,b1) )

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
        plane_marker       = pick['plane_marker']
        p_indices, p_areas = np.unique(plane_marker, return_counts=True)
        p_areas            = dict( zip(p_indices.tolist(), p_areas.tolist()) )
        gt_marker            = pick['marker']
        gt_indices, gt_areas = np.unique(gt_marker, return_counts=True)
        gt_areas             = dict( zip(gt_indices.tolist(), gt_areas.tolist()) )
        plane2coeff = pick['plane2coeff']
        normalized_pcenters = pick['normalized_pcenters']

        plane2gidx   = pick['plane2marker']
        gidx2pindices = {}
        for pidx, gidx in plane2gidx.items():
            gt_area = gt_areas[gidx]
            p_area  = p_areas[pidx]
            if not pidx in plane2coeff:
                continue
            if not gidx in gidx2pindices:
                gidx2pindices[gidx] = []
            gidx2pindices[gidx].append( (pidx,float(p_area)) )
        for gidx, pidx_areas in gidx2pindices.items():
            sorted_list = sorted(pidx_areas, reverse=True, key=lambda x: x[1])
            sum_area = 0.
            for _, area in sorted_list:
                sum_area += area
            gidx2pindices[gidx] = list(map(lambda x:  (x[0], x[1] / sum_area), sorted_list))

        w,h = float(gt_marker.shape[1]), float(gt_marker.shape[0])
        dst = GetColoredLabel(plane_marker)
        for gidx in gt_indices:
            if gidx == 0:
                continue
            ncp = pick['normalized_centers'][gidx]
            ncp = np.array([ncp[0],ncp[1],1.])
            ncp /= np.linalg.norm(ncp)
            th1 = np.arccos( ncp.dot(np.array([0.,0.,1.])) )
            th1 = np.rad2deg(th1)

            th_incidence = 0.
            for pidx,p_area in gidx2pindices[gidx]:
                #if p_area < .1: # TODO .1, .3
                #    break
                ncp = normalized_pcenters[pidx]
                vec0 = -np.array([ncp[0],ncp[1],1.])
                vec0 /= np.linalg.norm(vec0)

                coeff0 = plane2coeff[pidx] # ax + by + cz + d = 0
                nvec   = np.array(coeff0[:3])
                th = np.arccos( nvec.dot(vec0) )
                th = np.rad2deg(th)
                #th_incidence = max(th, th_incidence)
                th_incidence += p_area*th

            indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
            oblique_array[indicies] = th_incidence
            #oblique_array[indicies] = oblique
        #cv2.imshow("dst", dst)
        #if ord('q') == cv2.waitKey():
        #    exit(1)
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

def GetMargin(eval_data, picks, normalized):
    margin_array = np.repeat(-1.,eval_data.shape)
    minwidth_array = np.repeat(-1.,eval_data.shape)
    for base, pick in picks.items():
        gt_marker = pick['marker']
        gindices = np.unique(gt_marker)
        for gidx in gindices:
            if gidx == 0:
                continue
            indicies, = np.where(np.logical_and(eval_data['base']==base,eval_data['gidx']==gidx))
            if normalized:
                normalized_minwidth = pick['normalized_minwidths'][gidx]
                normalized_margin   = pick['normalized_margins'][gidx]
                minwidth_array[indicies] = normalized_minwidth
                margin_array[indicies]   = normalized_margin
            else:
                minwidth = pick['minwidths'][gidx]
                margin   = pick['margins'][gidx]
                minwidth_array[indicies] = minwidth
                margin_array[indicies]   = margin
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

latest_pick = (None,None)
def LabelClicked(event,eval_dir,picks):
    global latest_pick
    if latest_pick == (event.artist, event.mouseevent.button):
        return
    latest_pick = (event.artist, event.mouseevent.button)
    if event.mouseevent.button == 1: # Left click
        print("Stop for over min_iou %.3f, colorize in bound"%event.artist.min_iou)
        show_cases = event.artist.sucess
    else:
        print("Stop for less min_iou %.3f, colorize in bound"%event.artist.min_iou)
        show_cases = ~event.artist.sucess
    ShowCases(eval_dir, picks, event.artist.eval_data[show_cases],
            event.artist.ydata_name, event.artist.valid[show_cases])
    return

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
                fontsize=FONT_SIZE-2, ha='center', va=va) #, bbox=dict(boxstyle='square,pad=.3'))
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

def GetThickness(eval_data):
    thickness_array = np.repeat(-1.,eval_data.shape)
    for i,b1 in enumerate(eval_data['predbox']):
        thickness_array[i] = b1.scale[2]
    return thickness_array

def PlotLengthOblique(picks, eval_data_allmethod, tags, min_iou,
        show_fig=True):
    ourobb_data = eval_data_allmethod[eval_data_allmethod['method']=='myobb']
    assert(len(ourobb_data)>0)
    mvbb_data = eval_data_allmethod[eval_data_allmethod['method']=='mvbb']
    if len(mvbb_data) == 0:
        rospy.logwarn("No samples for MVBB test")
    ransac_data = eval_data_allmethod[eval_data_allmethod['method']=='ransac']
    if len(ransac_data) == 0:
        rospy.logwarn("No samples for RANSAC test")

    valid = logical_ands([tags=='',
                          ourobb_data['iou']>min_iou,
                          ourobb_data['valid_obb']
                          ])
    ourobb_data = ourobb_data[valid]
    minmargin=10.
    thresh_thick = 0.05
    datas = Od()
    if len(ourobb_data)>0:
        margin, minwdith = GetMargin(ourobb_data, picks, normalized=False)
        ourobb_data = ourobb_data[margin > minmargin]
        datas['OBB for all']                  = ourobb_data
        datas['OBB for observable side']      = ourobb_data[GetThickness(ourobb_data) > thresh_thick]
        datas['OBB for unobservable side']    = ourobb_data[GetThickness(ourobb_data) < thresh_thick]
    if len(mvbb_data)>0:
        margin, minwdith = GetMargin(mvbb_data, picks, normalized=False)
        mvbb_data = mvbb_data[margin > minmargin]
        datas['MVBB for observable side']     = mvbb_data[GetThickness(mvbb_data) > thresh_thick]
        datas['MVBB for unobservable side']   = mvbb_data[GetThickness(mvbb_data) < thresh_thick]
    if len(ransac_data)>0:
        margin, minwdith = GetMargin(ransac_data, picks, normalized=False)
        ransac_data = ransac_data[margin > minmargin]
        datas['RANSAC for observable side']   = ransac_data[GetThickness(ransac_data) > thresh_thick]
        datas['RANSAC for unobservable side'] = ransac_data[GetThickness(ransac_data) < thresh_thick]

    error_names = ['trans_err', 'max_wh_err', 'deg_err']
    rows = []
    for data_name, data in datas.items():
        row = [data_name]
        for err_name in error_names:
            errors = data[err_name]
            for eval_type in ['Median', 'MAE']:
                if eval_type == 'Median':
                    val = np.median(errors)
                else:
                    val = np.sum(np.abs(errors)) / float(len(errors))
                row.append(val)
        rows.append(row)
    # ref : https://pyhdust.readthedocs.io/tabulate.html
    table = tabulate(rows, tablefmt="latex",
            floatfmt=(None,'.3f', '.3f', '.3f','.3f','.2f','.2f') )
    print(table)

    if not show_fig:
        return

    datas = Od()
    if len(ourobb_data)>0:
        datas['OBB']                          = ourobb_data # for all cases
    if len(mvbb_data)>0:
        datas['MVBB for observable side']     = mvbb_data[GetThickness(mvbb_data) > thresh_thick]
    if len(ransac_data)>0:
        datas['RANSAC for unobservable side'] = ransac_data[GetThickness(ransac_data) < thresh_thick]
    min_length = .1
    fig = plt.figure(figsize=(8,6), dpi=DPI)
    fig.subplots_adjust(**FIG_SUBPLOT_ADJUST)
    axes = Od()
    axes['deg_err']    = fig.add_subplot(3,1,3)
    axes['trans_err']  = fig.add_subplot(3,1,1)
    axes['max_wh_err'] = fig.add_subplot(3,1,2)

    for err_name, ax in axes.items():
        if err_name == 'deg_err':
            n_bins, step = 5, 10.
            unit, _format = '[deg]', '%.f~%.f'
            ax.set_title('Oblique error-PDF',fontsize=7)#.set_position( (.5, 0.))
        elif err_name == 'trans_err':
            n_bins, step = 5, 0.05
            unit, _format = '[m]', '%.3f~%.3f'
            ax.set_title('Trans error-PDF',fontsize=7)#.set_position( (.5, 1.42))
        elif err_name == 'max_wh_err':
            n_bins, step = 5, 0.05
            unit, _format = '[cm]', '%.1f~%.1f'
            ax.set_title('Size error-PDF',fontsize=7)#.set_position( (.5, 1.42))

        min_max = [0., n_bins*step]
        x = np.arange(n_bins)
        max_bound = 0.
        nbar = len(datas)
        width = 1. / float(nbar) - .05
        x = np.arange(n_bins)
        offset = float(nbar-1)*width/2.

        for i, (method, data) in enumerate(datas.items()):
            lengths = GetMinLength(data, picks)
            values = data[lengths > min_length][err_name]
            tp_hist,  bins = np.histogram(values, n_bins, min_max)
            tp_hist = tp_hist.astype(float) / np.sum(tp_hist).astype(float)
            tp_hist[tp_hist==0.] = 1e-10 # For no missing label
            rects = ax.bar(x-offset, width=width, height=tp_hist, alpha=.5,label=method)
            LabelHeight(ax,rects)
            offset -=width
            for bound, y in zip(np.flip(bins[1:]), np.flip(tp_hist)):
                max_bound = max(bound/step,max_bound)
                if y > 0.:
                    break
        #ax.set_xlim(offset, max_bound)
        #ax.set_xlim(m)
        ax.set_xlabel(unit, fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
        ax.xaxis.set_label_coords(**XLABEL_COORD)
        ax.set_ylabel('Probability', fontsize=FONT_SIZE)
        ax.legend(loc='upper right', fontsize=FONT_SIZE)
        xlabels = []
        for i in range(n_bins):
            msg = _format%(bins[i],bins[i+1])
            xlabels.append(msg)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=0.,fontsize=FONT_SIZE)
    fig.tight_layout()
    return fig, axes

def Plot2dEval(eval_data, picks, ydata_name, valid, ax, min_ious, num_bins, min_max,
        unit_str, _format,
        show_underseg=False, show_overseg=False, show_sample=False):
    try:
        ydata = eval_data[ydata_name]
    except:
        import pdb; pdb.set_trace()
    la = np.logical_and
    n_hist , bound    = np.histogram(ydata[valid], num_bins,min_max)

    #if show_overseg:
    #    cond.append( ~eval_data['overseg'] )
    #if show_underseg:
    #    cond.append( ~eval_data['underseg'] )

    tp_hists = Od()
    for min_iou in min_ious:
        cond = [eval_data['iou']>min_iou, valid]
        tp_hists[min_iou], _       = np.histogram(ydata[la.reduce(cond)], num_bins, min_max)

    underseg_hist , _ = np.histogram(ydata[la(eval_data['underseg'],valid)], num_bins,min_max)
    overseg_hist , _  = np.histogram(ydata[la(eval_data['overseg'],valid)], num_bins,min_max)
    no_samples = n_hist==0
    n_hist[no_samples] = 1 # To prevent divide by zero
    if show_underseg:
        if underseg_hist.sum()==0:
            show_underseg = False
    if show_overseg:
        if overseg_hist.sum()==0:
            show_overseg = False
    for min_iou, tp_hist in tp_hists.items():
        tp_hists[min_iou] = tp_hist.astype(float) / n_hist.astype(float)
    underseg_hist = underseg_hist.astype(float) / n_hist.astype(float)
    overseg_hist  = overseg_hist.astype(float) / n_hist.astype(float)
    n_hist[no_samples] = 0
    x = np.arange(num_bins)

    hists = []
    for min_iou, tp_hist in tp_hists.items():
        ap_label = 'AP(IoU >%.1f)'%min_iou
        hists.append( (tp_hist,ap_label) )

    if show_underseg:
        hists.append( (underseg_hist,'$p(\mathrm{under})$') )
    if show_overseg:
        hists.append( (overseg_hist,'$p(\mathrm{over})$') )

    b, r = float(len(hists)), .3
    width = 1. / (r*(b+1.)+b)
    offset = r * width

    for i, (hist, label) in enumerate(hists):
        hist[hist==0.] = 1e-10 # For no missing label
        dx = offset + float(i)*(width+offset)
        rects = ax.bar(x + dx , width=width, height=hist, alpha=.4, align='edge', label=label)
        LabelHeight(ax, rects)
        if i >= len(min_ious):
            continue
        min_iou = min_ious[i]
        for ix, artist in enumerate(rects):
            artist.ix = ix
            cond = la.reduce([ydata>bound[ix], ydata<bound[ix+1] ])#, eval_data['iou']<min_iou])
            artist.eval_data  = eval_data[cond]
            artist.valid      = valid[cond]
            artist.sucess     = (eval_data['iou']>min_iou)[cond]
            artist.min_iou    = min_iou
            artist.ydata_name = ydata_name
            artist.set_picker(True)

    x = np.arange(len(x)+1)
    xlabels = []
    for i in range(len(bound)):
        msg = _format%(bound[i])
        if i < len(n_hist) and show_sample:
            msg += '\n%10d'%n_hist[i]
        xlabels.append(msg)
    ax.set_xlim(x[0],  x[-1])
    ax.set_xlabel('%s'%unit_str,rotation=0, fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    ax.xaxis.set_label_coords(**XLABEL_COORD)
    ax.yaxis.set_label_coords(-0.08, 1.)

    #ax.set_yticks([0.,50.,100.])
    ax.set_yticks([])

    if len(hists) > 1:
        ax.legend(ncol=len(hists),**LEGNED_ARGS)
    else:
        ax.set_ylabel(ap_label,rotation=0, fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)

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
        xdata, valid, unit_str, _format,
        show_sample=True,
        num_bins=3, min_max=(.5, 3.) ):
    '''
    * 'Bar' Median error
    * 상자 물리적크기,이미지크기, 2D IoU, 이미지 위치. ->  t_err, deg_err, s_err
    * ref : https://matplotlib.org/stable/gallery/axes_grid1/parasite_simple.html#sphx-glr-gallery-axes-grid1-parasite-simple-py
    '''
    la = np.logical_and
    ax_deg = ax.twinx()
    ax.set_ylabel('[cm]', fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    ax_deg.set_ylabel('[deg]', fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
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


    # https://stackoverflow.com/questions/11774822/matplotlib-histogram-with-errorbars
    #yvalues = rmses
    if ytype.lower() == 'median':
        yvalues = medians
    elif ytype.lower() == 'mae':
        yvalues = maes
    #for k, v in yvalues.items():
    #    if v == 0.:
    #        yvalues[k] = 1e-10

    hists = []
    hists.append( (yvalues['trans_err'],'trans error','%.2f') )
    hists.append( (yvalues['max_wh_err'], 'size error','%.2f') )
    hists.append( (yvalues['deg_err'], 'rotation error','%.1f') )

    b, r = float(len(hists)), .2
    width = 1. / (r*(b+1.)+b)
    offset = r * width

    x = np.arange(num_bins).astype(float)
    for i, (hist, label, height_form) in enumerate(hists):
        hist[hist==0.] = 1e-10 # For no missing label
        dx = offset + float(i)*(width+offset)
        rects = ax.bar(x + dx , width=width, height=hist, alpha=.4, align='edge', label=label)
        LabelHeight(ax, rects, form=height_form)

    x = np.arange(len(x)+1)
    xlabels = []
    for i in range(len(bound)):
        msg = _format%(bound[i])
        if i < len(n_hist) and show_sample:
            msg += '\n%10d'%n_hist[i]
        xlabels.append(msg)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=FONT_SIZE)
    ax.set_xlabel('%s'%unit_str,rotation=0, fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
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
    ax.legend(ncol=len(hists),**LEGNED_ARGS)

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

    ax.set_xlabel('Case',rotation=0, fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
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
        try:
            b0 = marker2box(gt_obb)
        except:
            import pdb; pdb.set_trace()
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
        output23dlist.append( output+(True, t_err, deg_err, max_wh_err, b0,b1) )
    return output23dlist

def UnprojectIntensity(rect_intensity, rect_depth, rect_K, rect_D, frame_id):
    rect_rgb = cv2.cvtColor(rect_intensity, cv2.COLOR_GRAY2BGR)
    marker = np.ones_like(rect_depth, dtype=np.int32)
    xyzrgb, _ = UnprojectPointscloud(rect_rgb,rect_depth,marker,rect_K,rect_D,
            leaf_xy=0.01,leaf_z=0.01, do_ecufilter=False)
    # Define the fields of the point cloud message
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('r', 12, PointField.FLOAT32, 1),
              PointField('g', 16, PointField.FLOAT32, 1),
              PointField('b', 20, PointField.FLOAT32, 1)]
    header = rospy.Header()
    header.frame_id = frame_id
    pointcloud_msg = pc2.create_cloud(header, fields, xyzrgb)
    return pointcloud_msg

def CaptureScreen(rect_rgb, rect_intensity, obb_resp): #, vismarker_msg):
    # Capture the screenshot of the region
    x1,y1,w,h = 650, 200, 800, 700
    capture = pyautogui.screenshot(region=(x1, y1, w,h))
    # Convert the screenshot to a numpy array
    capture = np.array(capture)
    # Convert the color format from RGB to BGR (which is used by cv2)
    capture = cv2.cvtColor(capture, cv2.COLOR_RGB2BGR)

    height, width = rect_rgb.shape[:2]
    rgb = np.frombuffer(rect_rgb.data, dtype=np.uint8).reshape(height, width,-1)
    intensity = np.frombuffer(rect_intensity.data, dtype=np.uint8).reshape(height, width,-1)
    outline = np.frombuffer(obb_resp.filtered_outline.data, dtype=np.uint8)\
            .reshape(height, width)
    marker = np.frombuffer(obb_resp.marker.data, dtype=np.int32).reshape(height,width)
    return {'marker':marker, 'outline':outline, 'capture':capture, 'rgb':rgb,
            'intensity':intensity}

class ImageSubscriber:
    def __init__(self, topic):
        self.img = None

        # Subscribe to the image topic
        self.sub = rospy.Subscriber(topic, Image, self.image_callback)
        self.topic = topic

    def image_callback(self, msg):
        self.img = msg

def perform_test(eval_dir, gt_files,fn_evaldata, methods=['myobb']):
    rospy.loginfo("Waiting PredictEdge")
    rospy.wait_for_service('~PredictEdge')
    predict_edge = rospy.ServiceProxy('~PredictEdge', ros_unet.srv.ComputeEdge)
    rospy.loginfo("Waiting ComputeObb")
    rospy.wait_for_service('~ComputeObb')
    compute_obb = rospy.ServiceProxy('~ComputeObb', ros_unet.srv.ComputeObb)
    bridge = CvBridge()

    rospy.loginfo("Waiting GetBg")
    rospy.wait_for_service('~GetBg')
    get_bg = rospy.ServiceProxy('~GetBg', ros_unet.srv.GetBg)

    rospy.loginfo("Waiting Cgal")
    rospy.wait_for_service('~Cgal/ComputeObb')
    cgal_compute_obb = rospy.ServiceProxy('~Cgal/ComputeObb', ros_unet.srv.ComputePoints2Obb)
    rospy.loginfo("Waiting Ransac")
    rospy.wait_for_service('~Ransac/ComputeObb')
    ransac_compute_obb = rospy.ServiceProxy('~Ransac/ComputeObb', ros_unet.srv.ComputePoints2Obb)

    pub_gt_obb = rospy.Publisher("~gt_obb", MarkerArray, queue_size=-1)
    pub_gt_pose = rospy.Publisher("~gt_pose", PoseArray, queue_size=1)
    pub_xyzi = rospy.Publisher("~xyzi", PointCloud2, queue_size=5)

    pub_rgb = rospy.Publisher("~rgb", Image, queue_size=-1)
    pkg_dir = get_pkg_dir()

    nframe_per_scene = rospy.get_param('~nframe_per_scene',-1)
    assert(nframe_per_scene > 0)

    # output23dlist.append( ~
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
            ('gtbox',object),
            ('predbox',object),
            ]

    eval_data = None
    br = tf2_ros.StaticTransformBroadcaster()

    elapsed_times=Od()
    for m in methods:
        elapsed_times[m] = []
    for i_file, gt_fn in enumerate(gt_files):
        pick = get_pick(gt_fn)
        base = osp.splitext(osp.basename(pick['rosbag_fn']))[0]
        bag, set_depth, set_rgb, set_imu, topic2cam, rgb_topics, depth_topics, imu_topics, rgb_msgs, depth_msgs, imu_msgs,\
                info_msgs, rect_info_msgs, mx, my, fx, fy = \
                get_topics(bridge,pkg_dir,gt_fn, pick)
        #Tfwc = convert2tf(Twc)
        gt_obbs = {}
        for obb in pick['obbs']:
            gt_obbs[obb['id']] = obb

        gt_obb_markers = None

        eval_scene, nframe = [], 0
        transforms = {}
        for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
            for tf in msg.transforms:
                transforms['%s-%s'%(tf.header.frame_id, tf.child_frame_id)] = tf

        imu_topic_exists = True
        try:
            _, msg, _ = bag.read_messages(topics=imu_topics.values()).next()
        except:
            imu_msg = None
            imu_topic_exists = False

        rect_intensity_topic = '/cam0/helios2/intensity_rect'
        try:
            _, rect_intensity_msg, _ = bag.read_messages(topics=[rect_intensity_topic]).next()
            rect_intensity = np.frombuffer(rect_intensity_msg.data, dtype=np.uint8)\
                    .reshape(rect_intensity_msg.height, rect_intensity_msg.width)
        except:
            pass


        for topic, msg, t in bag.read_messages(topics=rgb_topics.values()\
                                                      +depth_topics.values()\
                                                      +imu_topics.values() ):
            cam_id = topic2cam[topic]
            if topic in set_depth:
                depth_msgs[cam_id] = msg
            elif topic in set_rgb:
                rgb_msgs[cam_id] = msg
            elif topic in set_imu:
                imu_msgs[cam_id] = msg
            rgb_msg, depth_msg = rgb_msgs[cam_id], depth_msgs[cam_id]
            if imu_topic_exists:
                imu_msg = imu_msgs[cam_id]
            info_msg = info_msgs[cam_id]
            rect_info_msg = rect_info_msgs[cam_id]
            if depth_msg is None or rgb_msg is None:
                continue
            if imu_msg is None and imu_topic_exists:
                continue
            if len(info_msg.D) == 0:
                rect_info_msg = info_msg
                rect_rgb_msg = rgb_msg
                rect_depth_msg = depth_msg
                rect_rgb = np.frombuffer(rect_rgb_msg.data, dtype=np.uint8)\
                        .reshape(rect_rgb_msg.height, rect_rgb_msg.width, -1)
            else:
                rect_info_msg, mx, my = get_rectification(info_msg)
                rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8)\
                            .reshape(rgb_msg.height, rgb_msg.width, -1)
                intensity = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                intensity_msg = bridge.cv2_to_imgmsg(intensity, encoding='mono8')
                intensity_msg.header.frame_id = rgb_msg.header.frame_id
                rect_rgb_msg, rect_depth_msg, rect_intensity_msg, rect_depth, rect_rgb, rect_intensity\
                        = rectify(rgb_msg, depth_msg, intensity_msg, mx, my, bridge)
            if rect_rgb_msg.header.frame_id == 'arena_camera':
                rect_rgb_msg.header.frame_id = rect_depth_msg.header.frame_id\
                        = rect_intensity_msg.header.frame_id\
                        = 'cam0_arena_camera'

            if pub_rgb.get_num_connections() > 0:
                pub_rgb.publish(rect_rgb_msg)
            rect_depth = bridge.imgmsg_to_cv2(rect_depth_msg, desired_encoding='passthrough')
            rect_K = np.array(rect_info_msg.K,np.float).reshape((3,3))
            rect_D = np.array(rect_info_msg.D,np.float).reshape((-1,))

            now = rospy.Time.now()
            tf_lists = []
            for k,v in transforms.items():
                if k  == 'cam0_depth_camera_link-cam0_imu_link':
                    continue
                tf = TransformStamped()
                tf.header.frame_id = v.header.frame_id
                tf.header.stamp = now
                tf.child_frame_id = v.child_frame_id
                tf.transform = v.transform
                tf_lists.append(tf)
            br.sendTransform(tf_lists)

            pointcloud_msg = UnprojectIntensity(rect_intensity, rect_depth, rect_K, rect_D, rect_depth_msg.header.frame_id)
            pub_xyzi.publish(pointcloud_msg)

            if gt_obb_markers is None:
                cam_frame_id = rgb_msgs[cam_id].header.frame_id
                gt_obb_poses, gt_obb_markers = VisualizeGt(gt_obbs, frame_id=cam_frame_id)
                a = Marker()
                a.action = Marker.DELETEALL
                for arr in [gt_obb_markers]: # ,gt_infos
                    arr.markers.append(a)
                    arr.markers.reverse()

            t0_myobb = time.time()
            edge_resp = predict_edge(rect_rgb_msg,rect_depth_msg, fx, fy)
            if imu_msg is None:
                bg_mask = np.zeros((rect_rgb_msg.height,rect_rgb_msg.width),dtype=np.int32)
                bg_mask = bridge.cv2_to_imgmsg(bg_mask,encoding='32SC1')
            else:
                bg_res = get_bg(rect_rgb_msg,rect_depth_msg,rect_info_msg,imu_msg)
                bg_mask = bg_res.p_mask
            try:
                obb_resp = compute_obb(rect_depth_msg, rect_intensity_msg, edge_resp.edge, rect_info_msg, std_msgs.msg.String(cam_id), bg_mask)
            except:
                import pdb; pdb.set_trace()
            t1_myobb = time.time()

            t = (time.time()-t0_myobb) / float(len(obb_resp.output.markers))
            elapsed_times['myobb'].append(t)

            if 'mvbb' in methods or 'ransac' in methods:
                '''
                # Comaprison for cgal obb
                * [x] collecting cases
                * [x] Cgal OBB marker array로 획득.
                * [x] deg_err 계산해서 반영.
                * [x] 2면이 보이는 instance만 따로 골라내기
                    * Cgal OBB둘다 일정 크기 이상의 깊이를 가지면 orientation 문제가 감지됨.
                        -> marker에 상자 두께가 관찰되는 상황이라 이야기하자.
                '''
                marker = pick['marker'].copy()
                dist = cv2.distanceTransform( (~pick['outline']).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
                marker[dist < 5.] = 0
                xyzrgb, labels = UnprojectPointscloud(rect_rgb,rect_depth,marker,rect_K,rect_D,
                        leaf_xy=0.01,leaf_z=0.01, do_ecufilter=False)
                xyz = xyzrgb[:,:3].reshape(-1,).tolist()
                labels = labels.reshape(-1,).tolist()
                resps = {}
                if 'mvbb' in methods:
                    t0 = time.time()
                    resps['mvbb'] = cgal_compute_obb(xyz, labels, rect_depth_msg.header.frame_id)
                    t = (time.time()-t0) / float(len(resps['mvbb'].output.markers))
                    elapsed_times['mvbb'].append(t)
                if 'ransac' in methods:
                    t0 = time.time()
                    resps['ransac'] = ransac_compute_obb(xyz, labels, rect_depth_msg.header.frame_id)
                    t = (time.time()-t0) / float(len(resps['ransac'].output.markers))
                    elapsed_times['ransac'].append(t)
                for method, resp in resps.items():
                    eval_23d = GetErrorOfMvbb(gt_obb_markers, resp)
                    for each in eval_23d:
                        eval_scene.append( (base,i_file,nframe,method)+each)

            if 'myobb' in methods:
                # strict rule : 논문미팅 지도에 따라, 조그만 조각이라도 instace 나누면 overseg 판정.
                eval_2d, dst = Evaluate2D(obb_resp, pick['marker'], rect_rgb, strict_rule=False)
                eval_23d, dst3d = Evaluate3D(pick, gt_obb_markers, obb_resp, eval_2d)
                for each in eval_23d:
                    eval_scene.append( (base,i_file,nframe,'myobb')+ each)
            pub_gt_obb.publish(gt_obb_markers)

            capture = CaptureScreen(rect_rgb,rect_intensity_msg,obb_resp)
            with open(osp.join(eval_dir,"capture_%d_%d.pick"%(i_file+1, nframe+1)),'wb') as f:
                pickle.dump(capture, f, protocol=2)
            fn = osp.join(eval_dir, 'capture_%d_%d_%s.png'%(i_file+1,nframe+1,base))
            cv2.imwrite(fn, capture['capture'])
            #cv2.imshow('capture', capture['capture'])

            fn = osp.join(eval_dir, 'frame_%d_%s_%04d.png'%(i_file+1,base,nframe+1) )
            cv2.imwrite(fn, dst)

            nframe += 1
            rospy.loginfo("Perform evaluation for %s, S[%d/%d], F[%d/%d]"\
                    %(base, i_file+1,len(gt_files), nframe,nframe_per_scene) )
            #depth_msg, rgb_msg = None, None
            depth_msgs[cam_id], rgb_msgs[cam_id] = None, None
            if nframe >= nframe_per_scene:
                break
        eval_scene = np.array(eval_scene, dtype)
        if eval_data is None:
            eval_data = eval_scene
        else:
            eval_data = np.hstack((eval_data,eval_scene) )
        dst = visualize_scene(pick,eval_scene)
        fn = osp.join(eval_dir, 'scene_%d_%s.png'%(i_file,base) )
        cv2.imwrite(fn, dst)

    cv2.destroyAllWindows()
    with open(fn_evaldata,'wb') as f:
        np.save(f, eval_data)
    fn = osp.join(eval_dir, 'elapsed_times.pick')
    with open(fn,'wb') as f:
        pickle.dump(elapsed_times, f, protocol=2)

    return eval_data, elapsed_times

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
    fig = ax.get_figure()
    ren = fig.canvas.get_renderer()
    items = [ax] # ax.get_window_extent need renderer to prevent crush
    items += ax.get_xticklabels()
    #items += ax.get_yticklabels()
    items.append(ax.xaxis.label)
    items.append(ax.yaxis.label)
    item = ax.get_legend()
    if item is not None:
        items.append(item)
    fig = ax.get_figure()
    bbox = Bbox.union([item.get_window_extent(renderer=ren).transformed(fig.dpi_scale_trans.inverted()) \
            for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)

def logical_ands(list_of_arrays):
    return np.logical_and.reduce(list_of_arrays)
    #valid = list_of_arrays[0]
    #for i, arr in enumerate(list_of_arrays):
    #    if i == 0:
    #        continue
    #    valid = np.logical_and(valid, arr)
    #return valid

def captures2video(eval_dir, captures):
    sorted_captures = []
    max_scene, max_frame = -1, -1
    for fn in captures:
        scene, frame = osp.splitext(osp.basename(fn))[0].split('_')[1:]
        scene, frame = int(scene), int(frame)
        sorted_captures.append( (scene,frame,fn) )
        max_scene, max_frame = max(max_scene, scene), max(max_frame, frame)
    sorted_captures = sorted(sorted_captures, key=lambda x: (x[0], x[1]))

    fps = 2.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify the codec for WMV format
    video_fn = osp.join(eval_dir,"output.mp4")
    video_writer = None
    font_face, font_scale, font_thick = cv2.FONT_HERSHEY_PLAIN, 2.,2
    wait = False
    for scene, frame, fn in sorted_captures:
        with open(fn,'rb') as f:
            pick = pickle.load(f)
        marker, outline, capture, _, intensity \
                = pick['marker'], pick['outline'], pick['capture'], pick['rgb'],pick['intensity']
        rgb = cv2.cvtColor(intensity, cv2.COLOR_GRAY2BGR)
        capture = cv2.resize(capture,(marker.shape[1],marker.shape[0]))

        msg = "#s %2d/%d, #f %2d/%d"%(scene,max_scene,frame,max_frame)
        w,h = cv2.getTextSize(msg, font_face,font_scale,font_thick)[0]
        vis_marker = GetColoredLabel(marker,text=False)
        vis_marker = cv2.addWeighted(rgb, .3, vis_marker, .7, 0.)

        bg = marker==0
        vis_marker[bg,:] = rgb[bg,:]

        #dst_outline = cv2.addWeighted(rgb, .3, dst_outline, .7, 0.)
        #vis_marker = cv2.addWeighted(vis_marker, .3, dst_outline, .7, 0.)

        #vis_marker[outline > 0, 2] = 255
        #vis_marker[outline > 0, :2] = 0
        vis_marker = vis_marker.astype(np.int32)
        vis_marker[outline > 0, :]  -= 50
        vis_marker[vis_marker>255]  = 255
        vis_marker[vis_marker<0]  = 0
        vis_marker = vis_marker.astype(np.uint8)

        #boundary = GetBoundary(marker,2)
        #vis_marker[boundary>0,:] = 0

        dst = np.hstack((vis_marker, capture))
        cv2.imwrite(osp.join(eval_dir,'capture_%d_%d.png'%(scene,frame)), dst)
        if wait:
            cv2.imshow("dst", dst)
            c = cv2.waitKey()
            if ord('q') == c:
                exit(1)
            elif ord('b') == c:
                wait = False

        cv2.putText(dst, msg,
                (5,5+h), font_face, font_scale, (255,255,255), font_thick)
        if video_writer is None:
            video_writer =  cv2.VideoWriter(video_fn, fourcc, fps,
                    (dst.shape[1], dst.shape[0]))
        video_writer.write(dst)
    video_writer.release()

def JoinStructuredArray(arr1, arr2):
    # Create a new dtype that includes fields from both arrays
    new_dtype = arr1.dtype.descr + arr2.dtype.descr

    # Create an empty array with the new dtype
    joined_array = np.empty(arr1.shape, dtype=new_dtype)

    # Assign values from arr1 and arr2 to the new array
    for field in arr1.dtype.fields:
        joined_array[field] = arr1[field]

    for field in arr2.dtype.fields:
        joined_array[field] = arr2[field]

    return joined_array


def ShowCases(eval_dir, picks, eval_data, ydata_name, valid):
    if len(eval_data) == 0:
        return
    eval_data = JoinStructuredArray(eval_data,valid.astype([('valid',valid.dtype)]) )
    samples = eval_data

    unique_scenes = np.sort(np.unique(samples[['base','sidx']]), order='sidx')
    quit_flag = False
    for each_scene in unique_scenes:
        if quit_flag:
            break
        sidx, base = each_scene[['sidx','base']]
        sidx_eq = samples['sidx']==sidx
        frames_at_scene = np.unique( samples[sidx_eq]['fidx'] )
        pick = picks[base] 
        obbs = {}
        for obb in pick['obbs']:
            obbs[obb['id']] = obb

        for fidx in frames_at_scene:
            fidx_eq = samples['fidx']==fidx
            fn = osp.join(eval_dir,'frame_%d_%s_%04d.png'%(sidx+1,base,fidx+1) )
            all_gindices = samples[ np.logical_and.reduce([sidx_eq, fidx_eq]) ]['gidx']
            gindices = np.unique(all_gindices)
            mask = []
            for gidx in gindices:
                mask.append(pick['marker']==gidx)
            mask = np.logical_or.reduce( mask )
            src = cv2.imread(fn)
            frame_dst = src[:,:mask.shape[1],:]
            gt_dst    = GetColoredLabel(pick['marker'],True)
            gray_dst = (pick['marker']>0).astype(np.uint8)*200
            gray_dst = cv2.cvtColor(gray_dst,cv2.COLOR_GRAY2BGR)
            dst      = gray_dst.copy()
            dst[mask,:] = gt_dst[mask,:]
            dst = np.hstack( (dst,frame_dst) )

            font_face, font_scale, font_thick = cv2.FONT_HERSHEY_PLAIN, 1.,1
            x,y,o = 5,5,5
            nvalid = 0
            for gidx in gindices:
                gidx_eq = samples['gidx']==gidx
                val = samples[np.logical_and.reduce([sidx_eq,fidx_eq,gidx_eq])]
                assert(len(val)==1)
                val = val[0]
                msg = 'g#%2d, p#%2d: '%(gidx, val['pidx'])
                msg += 'IoU=%.2f, '%val['iou']
                # TODO 이거 대신 JoinStructuredArray 결과물
                msg += '%s=%.3f, '%(ydata_name,val[ydata_name])
                msg += 'dist=%3.2f, '%val['distance']
                msg += 'min(w,h)=%2.1f, '%val['minwidth']
                msg += 'margin=%3.3f, '%val['margin']
                msg += 'oblique=%2.1f, '%val['oblique']
                y += o+cv2.getTextSize(msg, font_face,font_scale,font_thick)[0][1]
                if val['valid']:
                    nvalid += 1
                    color = (255,255,255)
                else:
                    color = (0,0,255)
                cv2.putText(dst, msg, (x,y), font_face, font_scale, color, font_thick)

            #cv2.imshow("src", GetColoredLabel(pick['marker'],True) )
            cv2.imshow("dst", dst)
            if nvalid > 0:
                t = -1
            else:
                t = 1
            quit_flag = ord('q') == cv2.waitKey(t)
            if quit_flag:
                break
    cv2.destroyWindow("dst")
    #cv2.destroyWindow("src")
    #cv2.destroyAllWindows()
    return

def test_evaluation(rosbag_subnames, show_sample):
    pkg_dir = get_pkg_dir()
    eval_dir = osp.join(pkg_dir, 'eval_%s'%rosbag_subnames[0])
    if not osp.exists(eval_dir):
        makedirs(eval_dir)
    fn_evaldata = osp.join(eval_dir,'eval_data.npy')
    rospy.loginfo("The output path:%s"%eval_dir)

    black_lists = set()
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
        black_lists.add(base)
    black_lists.add('helio_2023-03-04-14-02-02') # Icebox with unflat surface
    tags = tmp_tags

    gt_files = []
    for usage in rosbag_subnames:
        obbdatasetpath = osp.join(pkg_dir,'obb_dataset_%s'%usage,'*.pick')
        ls0 = glob2.glob(obbdatasetpath)
        ls = []
        for fn in ls0:
            base = osp.splitext(osp.basename(fn))[0]
            groups = re.findall("(.*)_(cam0|cam1)", base)[0]
            base = groups[0]
            if base in black_lists:
                #import pdb; pdb.set_trace()
                continue
            ls.append(fn)
        #import pdb; pdb.set_trace()
        gt_files += ls
    if not osp.exists(fn_evaldata):
        methods = rospy.get_param("~methods").split(',')
        eval_data_allmethod, elapsed_times = perform_test(eval_dir, gt_files, fn_evaldata, methods=methods)
    else:
        with open(fn_evaldata,'rb') as f:
            eval_data_allmethod = np.load(f, allow_pickle=True)
        if(osp.basename(eval_dir) == 'eval_230428'):
            # 너무 오래걸리는 ransac, mvbb obb 불러오기.
            with open(osp.join(eval_dir+'_ransac', 'eval_data.npy'),'rb') as f:
                eval_other = np.load(f, allow_pickle=True)
                l1 = eval_other['method']=='ransac'
                l2 = eval_other['method']=='mvbb'
                eval_other = eval_other[np.logical_or(l1,l2)]
        eval_data_allmethod = np.concatenate((eval_data_allmethod, eval_other), axis=0)
        fn = osp.join(eval_dir, 'elapsed_times.pick')
        with open(fn,'rb') as f:
            elapsed_times = pickle.load(f)

    captures = glob2.glob(osp.join(eval_dir,'capture_*_*.pick'),recursive=False)
    captures2video(eval_dir, captures)

    picks = {}
    for fn in gt_files:
        base = osp.splitext( osp.basename(fn) )[0]
        groups = re.findall("(.*)_(cam0|cam1)", base)[0]
        base = groups[0]
        with open(fn,'r') as f:
            picks[base] = pickle.load(f)

    la = np.logical_and
    eval_data = eval_data_allmethod[eval_data_allmethod['method']=='myobb']
    oblique = GetOblique(eval_data, picks)
    normalized = False
    margin, minwidth = GetMargin(eval_data, picks, normalized)
    distance = GetDistance(eval_data, picks)
    tags   = GetTags(eval_data, picks, tags)

    eval_data = JoinStructuredArray(eval_data, oblique.astype([('oblique',oblique.dtype)]) )
    eval_data = JoinStructuredArray(eval_data, margin.astype([('margin',margin.dtype)]) )
    eval_data = JoinStructuredArray(eval_data, minwidth.astype([('minwidth',minwidth.dtype)]) )
    eval_data = JoinStructuredArray(eval_data, distance.astype([('distance',distance.dtype)]) )
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
        voblique = oblique < 40.
        #import pdb; pdb.set_trace()
        if normalized:
            vminwidth = minwidth > 30./500. # 0.02
            vmargin = margin > 100./500. # 0.04
            margin_minmax = (20./500., 120./500.)
            n_margin = 5
            width_minmax = (2./500., 70./500.)
            n_width = 4
            img_unit = ''
            img_format = '%.2f'
        else:
            vminwidth = minwidth > 50.
            vmargin = margin > 70. # TODO 30, 100
            margin_minmax = (20., 100.)
            width_minmax = (20., 70.)
            n_margin = 5
            width_minmax = 2., 70.
            n_width = 4
            img_unit = '[pixel]'
            img_format = '%.f'
        vdistance = distance < 1.5
        limit_valid = True
        show_underseg, show_overseg = True,True
        min_ious = [.6, .7]

        if limit_valid:
            Plot2dEval(eval_data, picks, 'margin', logical_ands([valid,vminwidth,vdistance,voblique]),
                    axes[0], num_bins=n_margin, min_max=margin_minmax, unit_str=img_unit, _format=img_format,
                    min_ious=min_ious, show_underseg=show_underseg, show_overseg=show_overseg, show_sample=show_sample)

            Plot2dEval(eval_data, picks, 'minwidth',  logical_ands([valid,vmargin,vdistance,voblique]),
                    axes[1], num_bins=n_width, min_max=width_minmax, unit_str=img_unit, _format=img_format,
                    min_ious=min_ious, show_underseg=show_underseg, show_overseg=show_overseg, show_sample=show_sample)

            Plot2dEval(eval_data, picks, 'oblique',
                    logical_ands([valid,vminwidth,vmargin,vdistance]),
                    axes[2], num_bins=4, min_max=(0., 60.), unit_str='[deg]', _format='%.f',
                    min_ious=min_ious, show_underseg=show_underseg, show_overseg=show_overseg, show_sample=show_sample)

            Plot2dEval(eval_data, picks, 'distance', logical_ands([valid,vminwidth,vmargin,voblique]),
                    axes[3], num_bins=4, min_max=(1., 1.8), unit_str='[m]', _format='%.2f',
                    min_ious=min_ious, show_underseg=show_underseg, show_overseg=show_overseg, show_sample=show_sample)

        else:
            Plot2dEval(eval_data, picks, 'margin', logical_ands([valid,]),
                    axes[0], num_bins=n_margin, min_max=margin_minmax, unit_str=img_unit, _format=img_format,
                    min_ious=min_ious, show_underseg=show_underseg, show_overseg=show_overseg, show_sample=show_sample)

            Plot2dEval(eval_data, picks, 'minwidth',  logical_ands([valid,]),
                    axes[1], num_bins=n_width, min_max=width_minmax, unit_str=img_unit, _format=img_format,
                    min_ious=min_ious, show_underseg=show_underseg, show_overseg=show_overseg, show_sample=show_sample)

            Plot2dEval(eval_data, picks, 'oblique', logical_ands([valid,]),
                    axes[2], num_bins=5, min_max=(0., 50.), unit_str='[deg]', _format='%.f',
                    min_ious=min_ious, show_underseg=show_underseg, show_overseg=show_overseg, show_sample=show_sample)

            Plot2dEval(eval_data, picks, 'distance', logical_ands([valid,]),
                    axes[3], num_bins=3, min_max=(0.5, 2.5), unit_str='[m]', _format='%.2f',
                    min_ious=min_ious, show_underseg=show_underseg, show_overseg=show_overseg, show_sample=show_sample)

        fig.canvas.mpl_connect('pick_event', lambda event:LabelClicked(event, eval_dir,picks) )

        axes[0].set_title('Margin-AP',fontsize=7).set_position( (.5, 1.42))
        axes[1].set_title('min(w,h)-AP',fontsize=7).set_position( (.5, 1.42))
        axes[2].set_title('Oblique-AP',fontsize=7).set_position( (.5, 1.42))
        axes[3].set_title('Distance-AP',fontsize=7).set_position( (.5, 1.42))
        #if not show_sample:
        fig.savefig(osp.join(eval_dir,'test_margin_ap.svg'),   transparent=True, bbox_inches=full_extent(axes[0]))
        fig.savefig(osp.join(eval_dir,'test_minwidth_ap.svg'), transparent=True, bbox_inches=full_extent(axes[1]))
        fig.savefig(osp.join(eval_dir,'test_oblique_ap.svg'),  transparent=True, bbox_inches=full_extent(axes[2]))
        fig.savefig(osp.join(eval_dir,'test_distance_ap.svg'), transparent=True, bbox_inches=full_extent(axes[3]))

        # TODO min_iou?
        min_iou = min_ious[0]
        valid = tags==''
        valid = la(valid, eval_data['iou']>min_iou)
        valid = la(valid, ~eval_data['underseg'])
        valid = la(valid, ~eval_data['overseg'])
        for i, ytype in enumerate(['Median', 'MAE']):
            n0 = 5*i
            Plot3dEval(eval_data, axes[n0+5], ytype,
                    margin, logical_ands([valid,vminwidth,vdistance]), show_sample=show_sample,
                    unit_str=img_unit, _format=img_format, num_bins=5, min_max=margin_minmax )
            # 의미있는 경향은 안보임.
            Plot3dEval(eval_data, axes[n0+6], ytype,
                    minwidth, logical_ands([valid,vmargin,vdistance]), show_sample=show_sample,
                    unit_str=img_unit, _format=img_format, num_bins=7, min_max=width_minmax )
            Plot3dEval(eval_data, axes[n0+7], ytype,
                    oblique, logical_ands([valid,vmargin,vminwidth,vdistance]), show_sample=show_sample,
                    unit_str ='[deg]', _format='%.1f', num_bins=5, min_max=(0, 50.) )
            Plot3dEval(eval_data, axes[n0+8], ytype,
                    distance, logical_ands([valid,vmargin,vminwidth]), show_sample=show_sample,
                    unit_str ='[m]', _format='%.1f', num_bins=3, min_max=(.5, 3.) )
            axes[n0+5].set_title('Margin - Err %s'%ytype,fontsize=7).set_position( (.5, 1.42))
            axes[n0+6].set_title('min(w,h) - Err %s'%ytype,fontsize=7).set_position( (.5, 1.42))
            axes[n0+7].set_title('Oblique - Err %s'%ytype,fontsize=7).set_position( (.5, 1.42))
            axes[n0+8].set_title('Distance - Err %s'%ytype,fontsize=7).set_position( (.5, 1.42))

        show_fig = False
        # Only valid 2D segmentations are counted for 3D evaluation
        res = PlotLengthOblique(picks, eval_data_allmethod, tags, min_iou,
                show_fig=show_fig)
        if show_fig:
            fig, axes = res
            for err_name, ax in axes.items():
                fn = 'test_pdf_%s.svg'% err_name
                fig.savefig(osp.join(eval_dir,fn), bbox_inches=full_extent(ax), transparent=True)

        print("Etime per each OBB-----")
        for method, etime in elapsed_times.items():
            sec = np.mean(etime)
            print("%s : %.1f [msec]" %  (method, sec*1000.) )
        print("----------")

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
        eval_data, elapsed_times = perform_test(eval_dir, gt_files, fn_evaldata)
    else:
        with open(fn_evaldata,'rb') as f:
            eval_data = np.load(f, allow_pickle=True)
    captures = glob2.glob(osp.join(eval_dir,'capture_*_*.pick'),recursive=False)
    captures2video(eval_dir, captures)
    picks = {}
    for fn in gt_files:
        base = osp.splitext( osp.basename(fn) )[0]
        groups = re.findall("(.*)_(cam0|cam1)", base)[0]
        base = groups[0]
        with open(fn,'r') as f:
            picks[base] = pickle.load(f)
    tags = {}
    distance = GetDistance(eval_data, picks)
    margin, minwidth = GetMargin(eval_data, picks, normalized=False)
    valid    = margin>40.
    fig = plt.figure(1, figsize=FIG_SIZE, dpi=DPI)
    fig.subplots_adjust(**FIG_SUBPLOT_ADJUST)
    ax = fig.add_subplot(N_FIG[0],N_FIG[1],1)
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    #PlotDistanceAp(eval_data, picks, distance, valid, ax, min_iou=.5,
    #        num_bins=5,min_max=(1.,2.5), show_underseg=True, show_overseg=True)
    Plot2dEval(eval_data, picks, 'distance',  valid, ax,
            num_bins=5, min_max=(1.5, 2.5), unit_str='[m]', _format='%.2f',
            min_iou=.7, show_underseg=True,show_overseg=True,show_sample=True)
    fig.savefig(osp.join(eval_dir,'test_dist_ap.svg'), transparent=True, bbox_inches=full_extent(ax))
    #PlotEachScens(eval_data, picks, eval_dir, infotype='')
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
        eval_data, elapsed_times = perform_test(eval_dir, gt_files, fn_evaldata)
    else:
        with open(fn_evaldata,'rb') as f:
            eval_data = np.load(f, allow_pickle=True)
    captures = glob2.glob(osp.join(eval_dir,'capture_*_*.pick'),recursive=False)
    captures2video(eval_dir, captures)
    picks = {}
    for fn in gt_files:
        base = osp.splitext( osp.basename(fn) )[0]
        groups = re.findall("(.*)_(cam0|cam1)", base)[0]
        base = groups[0]
        with open(fn,'r') as f:
            picks[base] = pickle.load(f)
    tags = {}
    margin, minwidth = GetMargin(eval_data, picks, normalized=False)
    oblique = GetOblique(eval_data, picks)
    valid   = margin > 20./500. # 0.04
    fig = plt.figure(1, figsize=FIG_SIZE, dpi=DPI)
    fig.subplots_adjust(**FIG_SUBPLOT_ADJUST)
    ax = fig.add_subplot(N_FIG[0],N_FIG[1],1)
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    Plot2dEval(eval_data, picks, oblique,  valid, ax,
            num_bins=5, min_max=(0., 50.), unit_str='[deg]', _format='%.f',
            min_iou=.7, show_underseg=True,show_overseg=True)

    fig.savefig(osp.join(eval_dir,'test_oblique_ap.svg'), bbox_inches=full_extent(ax))
    #PlotEachScens(eval_data, picks, eval_dir, infotype='')
    return

if __name__=="__main__":
    rospy.init_node('ros_eval', anonymous=True)
    target = rospy.get_param('~target')
    show = int(rospy.get_param('~show'))
    if target=='test':
        rosbag_subnames = rospy.get_param("~rosbag_subnames").split(',')
        rosbag_subnames = [x for x in rosbag_subnames if x]
        #rosbag_subnames = ['test0523']
        test_evaluation(rosbag_subnames, show_sample=show)
    elif target=='dist':
        dist_evaluation()
    elif target=='oblique':
        oblique_evaluation()

    if show:
        plt.show(block=True)
