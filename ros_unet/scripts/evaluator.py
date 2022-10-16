#!/usr/bin/python2
#-*- coding:utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import *
from geometry_msgs.msg import Point, Quaternion, Vector3
from geometry_msgs.msg import PoseArray, Pose
import cv2

import pickle
from Objectron import box, iou
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as rotation_util

from unet.gen_obblabeling import ParseGroundTruth, ParseMarker
from os import path as osp

import matplotlib.pyplot as plt
from matplotlib import gridspec
#from mpl_toolkits.mplot3d import axes3d, Axes3D 
import glob2
import re
from tabulate import tabulate
from unet.util import GetColoredLabel
from unet_ext import GetNeighbors as _GetNeighbors

tp_iou = .5

def get_pkg_dir():
    return osp.abspath( osp.join(osp.dirname(__file__),'..') )

def GetMarkerCenters(given_marker):
    marker = given_marker.copy()
    marker[:,0] = 0
    marker[:,-1] = 0
    marker[0,:] = 0
    marker[-1,:] = 0

    centers = {}
    for marker_id in np.unique(marker):
        if marker_id == 0:
            continue
        part = marker==marker_id
        dist_part = cv2.distanceTransform( part.astype(np.uint8),
                distanceType=cv2.DIST_L2, maskSize=5)

        loc = np.unravel_index( np.argmax(dist_part,axis=None), marker.shape)
        centers[marker_id] = (loc[1],loc[0])

    #dst = GetColoredLabel(marker)
    #for marker_id, cp in centers.items():
    #    cv2.putText(dst, "%d" % marker_id, (cp[0],cp[1]), cv2.FONT_HERSHEY_SIMPLEX, 1., (0), 2)

    return centers

def VisualizeMarker(rgb, marker):
    dst = rgb.copy()
    pass

def FitAxis(b0, b1):
    rwb0 = rotation_util.from_dcm(b0.rotation)
    rwb1 = rotation_util.from_dcm(b1.rotation)
    Rwb1 = np.zeros_like(b1.rotation)
    axis_index = [2,2,2]
    for k0 in range(2): # Fit x, y axis.
        axis0 = b0.rotation[:,k0]
        max_abscos, optimal_cos, optimal_k1 = 0., 0.,  -1
        for k1 in range(3):
            if axis_index[k1] < 2:
                continue
            axis1 = b1.rotation[:,k1]
            cos = np.dot(axis0, axis1)
            if abs(cos) > max_abscos:
                max_abscos, optimal_cos, optimal_k1 = abs(cos), cos, k1
        Rwb1[:,k0] = np.sign(optimal_cos) * b1.rotation[:,optimal_k1]
        axis_index[optimal_k1] = k0
    Rwb1[:,2] = np.cross(Rwb1[:,0], Rwb1[:,1])

    whd = b1.scale[axis_index]
    w = box.Box()
    b = w.from_transformation( Rwb1, b1.translation, whd)
    if np.min(b.scale) == 0.:
        import pdb; pdb.set_trace()
        FitAxis(b0,b1)
    return b

def GetCorrespondenceMarker():
    correspondence = Marker()
    correspondence.type = Marker.LINE_LIST
    correspondence.header.frame_id = "robot"
    correspondence.pose.orientation.w = 1.
    correspondence.scale.x = 0.05
    correspondence.color.a = 1.
    correspondence.color.r = 1.
    return correspondence

def GetInfoMarker(obj, precision, recall, iou_val, index):
    info = Marker()
    info.text = "type=%d"%obj.type
    info.text = "pr=%.2f"%precision
    info.text += "\nre=%.2f"%recall
    info.text += "\nIoU=%.2f"%iou_val
    info.header.frame_id = "robot"
    info.type = Marker.TEXT_VIEW_FACING
    info.scale.z = 0.02
    # TODO
    #if IsOversegmentation(precision,recall):
    #    info.color.r = 1.
    #    info.color.g = info.color.b = 0.
    #    pass
    #elif IsUndersegmentation(precision,recall):
    #    info.color.b = 1.
    #    info.color.r = info.color.b = 0.
    #    pass
    #else:
    #    info.color.r = info.color.g = info.color.b = 1.

    info.color.a = 1.
    info.pose = obj.pose
    info.id = index
    return info

def GetBoxMarker(given_box, index):
    marker = box2marker(given_box)
    marker.id = index
    marker.color.a = 0.8
    marker.color.r = 1.
    marker.color.g = marker.color.b = 0.2
    marker.header.frame_id = "robot"
    return marker

def GetDepthOptimizedBox(target_box, ref_box, daxis):
    Rwb0 = target_box.rotation

    # World좌표계의 depth 축에 가장 가까운것.
    optimized_axis = np.argmax(np.abs(Rwb0[daxis,:]))

    # depth axis의 반대방향으로
    sign = -np.sign(Rwb0[daxis,optimized_axis])

    optimized_gt_whd = target_box.scale.copy()
    axis1 = np.argmax(np.abs(ref_box.rotation[daxis,:]))
    optimized_val = min(optimized_gt_whd[optimized_axis], ref_box.scale[axis1])
    optimized_gt_whd[optimized_axis] = optimized_val

    offset = np.zeros((3,) )
    offset[optimized_axis] = sign*0.5*(target_box.scale[optimized_axis]-optimized_val)
    dX = np.matmul(Rwb0,offset).reshape((-1,))
    optimized_gt_xyz = target_box.translation + dX
    w0 = box.Box()
    return w0.from_transformation(Rwb0, optimized_gt_xyz, optimized_gt_whd)

def GetErrors(indices0, pairs0to1, pair_infos, min_iou):
    trans_errors = []
    deg_errors = []
    for i0 in indices0:
        for i1 in pairs0to1[i0]:
            info =pair_infos[(i0,i1)]
            if info['iou'] > min_iou:
                cp_surf0, rwb0  = info['surf0']
                cp_surf1, rwb1  = info['surf1']
                t_err = np.linalg.norm( cp_surf1-cp_surf0 )
                trans_errors.append(t_err)
                dr = rwb1.inv()* rwb0
                deg_err = np.rad2deg( np.linalg.norm(dr.as_rotvec()) )
                #if deg_err > 45.:
                #    import pdb; pdb.set_trace()
                deg_errors.append(deg_err)
                break
    return trans_errors, deg_errors

def DrawOutline(cvgt, rgb):
    dst = rgb.copy()
    outline, _, ext_marker, _,_ = ParseMarker(cvgt, rgb)
    dst[outline,0] = 255
    dst[outline,1:] = 0
    return dst

class Evaluator:
    def __init__(self):
        self.scene_evals = {}
        self.frame_evals = {}
        self.n_frame = 0
        self.n_evaluate = 0

    def PutScene(self, fn, scene_eval):
        base = osp.splitext( osp.basename(fn) )[0]
        self.scene_evals[base] = scene_eval 

    def PutFrame(self, fn, frame_eval):
        frame_eval.frame_id = self.n_frame
        self.n_frame += 1

        base = osp.splitext( osp.basename(fn) )[0]

        if not self.frame_evals.has_key(base):
            self.frame_evals[base] = []
        self.frame_evals[base].append(frame_eval)
        return len(self.frame_evals[base])

    def Evaluate(self, eval_dir, gt_files, arr_frames=None, all_profiles=None, is_final=False):
        self.n_evaluate += 1
        if all_profiles is None:
            arr_frames, all_profiles = self.GetTables()

        tmp = {}
        for fn in gt_files:
            scene_base = osp.basename(fn).split('_cam0.pick')[0]
            f = open(fn,'r')
            tmp[scene_base] = pickle.load(f)
            f.close()
            # TODO load pickle?
        gt_files = tmp

        tmp, segments = {}, glob2.glob(osp.join(eval_dir,'screenshot','segment*.png'))
        for fn in segments:
            base = osp.basename(fn)
            frame_id, scene_base = re.findall('segment_0*(\d+)_(.*).png',base)[0]
            if not scene_base in tmp:
                tmp[scene_base] = []
            tmp[scene_base].append(fn)
        segments = tmp

        if is_final:
            DrawApChart(gt_files, segments, arr_frames, all_profiles)
            fig.suptitle('Evaluation', fontsize=16)
        plt.tight_layout(pad=3., w_pad=2., h_pad=3.0)
        return

    def DrawEachScene(self, screenshot_dir):
        fn=self.scene_evals.values()[0].pick['cvgt_fn']
        fn = osp.join(get_pkg_dir(),fn)
        cvgt = cv2.imread(fn)
        if cvgt is None:
            import pdb; pdb.set_trace()
        fontsize=10
        dpi = 100
        height, width = cvgt.shape[:2]
        fig = plt.figure(2, figsize=(8,5), dpi=dpi)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 

        for scene_i, (base, scene_eval) in enumerate(self.scene_evals.items()):
            fig.clf()
            K = scene_eval.pick['newK']
            bb_id = 20
            fn = osp.join(get_pkg_dir(),scene_eval.pick['cvgt_fn'])
            cvgt = cv2.imread(fn)
            rgb = scene_eval.pick['rgb']
            dst = DrawOutline(cvgt, rgb)

            col_labels = ['IoU 3d>%.1f'%tp_iou, 'over', 'under']
            colWidths = [0.3] * len(col_labels)
            row_labels, table_vals = [], []

            ax = plt.subplot(gs[0])
            ax.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), extent = [0,1,0,1],
                    aspect=float(height)/float(width) )
            plt.axis('off')
            plt.axis('equal')
            #plt.tight_layout(pad=0., w_pad=0, h_pad=0.)

            overlay = dst.copy()
            for gt_i, (gt_id, gt_obb) in enumerate( scene_eval.gt_obbs.items() ):
                gt_i += 1
                n_tp, n_overseg, n_underseg = 0, 0, 0
                n_frame = len(self.frame_evals[base])


                for frame_eval in self.frame_evals[base]:
                    #frame_id = frame_eval.frame_id
                    state = frame_eval.profiles['gt_states'][gt_id]
                    pred_id, iou2d = frame_eval.profiles['gt_max_iou2d'][gt_id]
                    pred_id, iou3d = frame_eval.profiles['gt_max_iou3d'][gt_id]
                    if iou3d > tp_iou:
                        n_tp += 1
                    else:
                        if 'overseg' in state:
                            n_overseg += 1
                        if 'underseg' in state:
                            n_underseg += 1
                row_labels.append('%d'%gt_i)
                row = [ float(v)/float(n_frame)*100 for v in [n_tp, n_overseg, n_underseg] ]
                #row = ['%.1f'%v+' %' for v in row ] 
                row = ['%.1f'%v for v in row ] 
                table_vals.append( row )

                pose_msg = Posetuple2Rosmsg(gt_obb['pose'])
                xyz_c, _ = GetSurfCenterPoint0(pose_msg, gt_obb['scale'], daxis=2)
                xyz_c /= xyz_c[2]
                uvz = np.matmul(K, xyz_c)
                text, font, font_scale, font_thickness = '#%d'%gt_i, cv2.FONT_HERSHEY_PLAIN, 2., 2
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_w, text_h = text_size
                uvz[0] -= .5*text_w
                uvz[1] += .5*text_h
                uvz[0] = max(5,          min(uvz[0], width-text_w ) )
                uvz[1] = max(text_h+10,  min(uvz[1], height-5) )
                uv = ( int(uvz[0]), int(uvz[1]) )
                if False: # Put Text
                    cv2.rectangle(overlay, (uv[0], uv[1]-5-text_h), (uv[0] + text_w, uv[1] + text_h), (200,200,200), -1)
                    cv2.putText(overlay, text, (uv[0],uv[1]), font, font_scale, (0,0,0), font_thickness)
                    cv2.putText(dst, text, (uv[0],uv[1]), font, font_scale, (0,0,0), font_thickness)
                ax.text(uvz[0]/width, 1.-uvz[1]/height, '#%d'%gt_i,
                        bbox=dict(fill=True, facecolor='white', alpha=.5, edgecolor='black', linewidth=0),
                        ha='left', va='bottom',
                        fontsize=fontsize)
            alpha=.8
            dst = cv2.addWeighted(overlay, alpha, dst, 1 - alpha, 0)
            fname = osp.join(screenshot_dir,'scene_%04d.png'%(scene_i+1))
            cv2.imwrite(fname, dst)
            fname = osp.join(screenshot_dir,'eval_%04d.txt'%(scene_i+1))
            ar = np.hstack((np.array(row_labels).reshape(-1,1), np.array(table_vals)))
            np.savetxt(fname, ar, fmt='%s', delimiter=',',
                    header=','.join(['ID']+col_labels), comments='')
            ax = plt.subplot(gs[1])
            the_table = ax.table(cellText=table_vals,
                                 colWidths=colWidths,
                                 rowLabels=row_labels,
                                 colLabels=col_labels,
                                 cellLoc='center right',
                                 loc='center left')
            the_table.scale(1,1.2)
            the_table.auto_set_font_size(True)
            #the_table.set_fontsize(fontsize)
            plt.axis('off')
            plt.axis('tight')
            plt.tight_layout(pad=0., w_pad=0, h_pad=0.)
            plt.savefig( osp.join(screenshot_dir, 'scenetable_%04d.svg'%(scene_i+1) ) )
            plt.show(block=False)
        return

    def GetTables(self):
        frames = []
        frame_evals= []
        for base, frame_evals_of_a_scene in self.frame_evals.items():
            frame_evals += frame_evals_of_a_scene
            pick = frame_evals_of_a_scene[0].scene_eval.pick
            for frame_eval in frame_evals_of_a_scene:
                etime = frame_eval.elapsed_time
                ninstance = len(frame_eval.scene_eval.gt_obbs)
                frames.append( (base, ninstance, etime) )
        arr_frames = np.array(frames, dtype=[('scene_basename',object),
                                             ('ninstance',int),
                                             ('elapsed_time',float)] )

        all_profiles = {} 
        all_profiles['pairs'] = {}
        all_profiles['gt_states'] = {}
        all_profiles['gt_properties'] = {}
        all_profiles['gt_max_iou2d'] = {}
        all_profiles['gt_max_iou3d'] = {}
        all_profiles['pred_max_iou2d'] = {}
        all_profiles['fp_list'] = {}

        for i_frame, frame in enumerate(frame_evals):
            all_profiles['fp_list'][i_frame] = frame.profiles['fp_list']
            for (gt_id,pred_id), info in frame.profiles['pairs'].items():
                all_profiles['pairs'][(i_frame,gt_id,pred_id)] = info
            for gt_id, v in frame.profiles['gt_states'].items():
                all_profiles['gt_states'][(i_frame,gt_id)] = v
            for gt_id, v in frame.profiles['gt_max_iou2d'].items():
                all_profiles['gt_max_iou2d'][(i_frame,gt_id)] = v
            for gt_id, v in frame.profiles['gt_max_iou3d'].items():
                all_profiles['gt_max_iou3d'][(i_frame,gt_id)] = v
            for gt_id, v in frame.profiles['gt_properties'].items():
                all_profiles['gt_properties'][(i_frame,gt_id)] = v
            for pred_id, v in frame.profiles['pred_max_iou2d'].items():
                all_profiles['pred_max_iou2d'][(i_frame,pred_id)] = v
        return arr_frames, all_profiles

class SceneEval:
    def __init__(self, pick, Twc, plane_c, max_z, cam_id):
        self.gt_marker = pick['marker']
        self.pick = pick
        # Convert OBB to world(frame_id='robot) coordinate.
        q,t = Twc.orientation, Twc.position
        Rwc = rotation_util.from_quat([q.x, q.y, q.z, q.w])
        twc = np.array((t.x,t.y,t.z))
        self.Twc = (Rwc, twc)

        gt_obbs = {}
        plane_c = np.array(plane_c)
        for i, obj in enumerate(pick['obbs']):
            pose_msg = Posetuple2Rosmsg(obj['pose'])
            surf_cp, _ = GetSurfCenterPoint0(pose_msg, obj['scale'], daxis=0)
            if surf_cp[2] > max_z:
                continue
            # Tcb -> Trc * Tcb
            pose_cb = obj['pose']
            tcb = np.array( pose_cb[:3]+(1.,)  ).reshape((-1,))
            d = plane_c.dot(tcb)
            if d < -.1:
                continue
            tcb = tcb[:3]
            Rcb = rotation_util.from_quat([pose_cb[4], pose_cb[5], pose_cb[6], pose_cb[3] ])
            Rwb = Rwc*Rcb
            twb = np.matmul(Rwc.as_dcm(),tcb) + twc
            q_xyzw = Rwb.as_quat()
            pose_wb = (twb[0], twb[1], twb[2], q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
            obj['pose_wb'] = pose_wb
            gt_obbs[obj['id']] = obj
        self.gt_obbs = gt_obbs

        ## Build KDTree for center position
        #centers = np.zeros( (len(gt_obbs),3) )
        #for i, gt_obb in enumerate(gt_obbs):
        #    xyz_qwxyz = gt_obb['pose_wb']
        #    centers[i,:] = np.array(xyz_qwxyz[:3]).reshape((1,3))
        #self.tree = KDTree(centers)

        self.pub_gt_obb = rospy.Publisher("~%s/gt_obb"%cam_id, MarkerArray, queue_size=-1)
        self.pub_gt_pose = rospy.Publisher("~%s/gt_pose"%cam_id, PoseArray, queue_size=1)
        self.pub_gt_info = rospy.Publisher("~%s/gt_info"%cam_id, MarkerArray, queue_size=1)


    def pubGtObb(self):
        center_poses0, obj_array0 = VisualizeGt(self.gt_obbs)
        gt_info = MarkerArray()
        for obj in obj_array0.markers:
            info = Marker()
            info.id = obj.id
            info.type = Marker.TEXT_VIEW_FACING
            info.text = "id=%d"%obj.id
            info.scale.z = 0.04
            info.color.r = info.color.g = info.color.b = 1.
            info.color.a = 1.
            info.header.frame_id = "robot"
            info.pose = obj.pose
            gt_info.markers.append(info)

        a = Marker()
        a.action = Marker.DELETEALL
        for arr in [obj_array0, gt_info]:
            arr.markers.append(a)
            arr.markers.reverse()

        rate = rospy.Rate(hz=1)
        while self.pub_gt_obb.get_num_connections() < 0:
            print("No subscriber for %s"%self.pub_gt_obb.name )
            rate.sleep()

        for i in range(1):
            rate.sleep()
            self.pub_gt_obb.publish(obj_array0)
            self.pub_gt_info.publish(gt_info)
            self.pub_gt_pose.publish(center_poses0)


class FrameEval:
    def __init__(self, scene_eval, cam_id, elapsed_time, verbose=True):
        self.elapsed_time = elapsed_time
        self.scene_eval = scene_eval

        if verbose:
            self.pub_infos = rospy.Publisher("~%s/info"%cam_id, MarkerArray, queue_size=1)
            self.pub_correspondence = rospy.Publisher("~%s/correspondence"%cam_id, Marker, queue_size=1)
            self.pub_marker_optmized_gt = rospy.Publisher("~%s/optimized_gt"%cam_id, MarkerArray, queue_size=1)
            self.pub_marker_converted_pred = rospy.Publisher("~%s/marker_converted_pred"%cam_id, MarkerArray, queue_size=1)

    def Evaluate2D(self, pred_marker):
        gt_marker = self.scene_eval.gt_marker
        # TODO Move below expandd to  ...
        #dist, gt_marker = cv2.distanceTransformWithLabels( (gt_marker==0).astype(np.uint8),
        #        distanceType=cv2.DIST_L2, maskSize=5)
        #gt_marker[dist > 7.] = 0
        gt_pred = np.stack((gt_marker, pred_marker), axis=2)
        # ref : https://stackoverflow.com/questions/24780697/numpy-unique-list-of-colors-in-the-image
        pair_marker, counts = np.unique(gt_pred.reshape(-1,2),axis=0, return_counts=True)
        pairs_counts = sorted(zip(pair_marker,counts), key=lambda x:x[1],reverse=True)

        gt_indices, gt_areas = np.unique(gt_marker, return_counts=True)
        gt_areas = dict( zip(gt_indices.tolist(), gt_areas.tolist()) )

        pred_indices, pred_areas = np.unique(pred_marker, return_counts=True)
        pred_areas = dict( zip(pred_indices.tolist(), pred_areas.tolist()) )

        lists = []
        for pair, count in pairs_counts:
            gt_id, pred_id = pair.tolist()
            lists.append( (gt_id, pred_id, count) )
        arr = np.array(lists, dtype=[('gt',int),('pred',int),('area',int)])
        arr_nonbg = arr[np.logical_and(arr['gt']>0, arr['pred']>0) ]

        #print(tabulate(arr_nonbg, arr_nonbg.dtype.names))
        # 1) Segment states
        pairs = {}
        gt_states = {}
        gt_max_iou2d = {}
        gt_properties = {}
        pred_max_iou2d = {}
        fp_list = set()

        (rwc, twc) = self.scene_eval.Twc
        rcw , tcw = rwc.inv(), -np.matmul(rwc.inv().as_dcm(), twc)

        center_poses0, obj_array0 = VisualizeGt(self.scene_eval.gt_obbs)
        markers0 = {}
        for marker in obj_array0.markers:
            markers0[marker.id] = marker

        for pred_id in pred_indices:
            if pred_id == 0:
                continue
            correspond = arr[arr['pred']==pred_id]
            correspond = np.sort(correspond, order='area')
            greatest = correspond[-1]['gt']
            if greatest > 0:
                continue
            #print(tabulate(correspond, correspond.dtype.names) )
            #import pdb; pdb.set_trace()
            fp_list.add(pred_id)

        for gt_id, gt_area in gt_areas.items():
            if gt_id == 0:
                continue
            gt_states[gt_id] = set()
            gt_max_iou2d[gt_id] = (-1, 0.)
            correspond = arr_nonbg[ arr_nonbg['gt']==gt_id ]
            occupies = []
            for pred_id, intersection in zip(correspond['pred'], correspond['area']):
                pred_area = pred_areas[pred_id]
                iou2d       = float(intersection) / float(gt_area + pred_area - intersection)
                recall2d    = float(intersection) / float(gt_area)
                precision2d = float(intersection) / float(pred_area)
                if iou2d > gt_max_iou2d[gt_id][1]:
                    gt_max_iou2d[gt_id] = (pred_id, iou2d)
                if  precision2d > .5:
                    occupies.append( (pred_id, iou2d, recall2d, precision2d) )
            if len(occupies) > 1:
                for pred_id, iou2d, recall2d, precision2d in occupies:
                    pairs[(gt_id, pred_id)] = {'state':'overseg',
                            'iou2d':iou2d, 'recall2d':recall2d, 'precision2d':precision2d }
                    gt_states[gt_id].add('overseg')

            # gt_properties
            obj0 = markers0[gt_id]
            b0 = marker2box(obj0)
            rwb0 = rotation_util.from_dcm(b0.rotation)

            daxis = 0
            cp_surf0, _ = GetSurfCenterPoint(obj0,daxis)

            # Center 'p'oint of ground truth on 'c'amera frame.
            twp0 = cp_surf0
            tcp0 = np.matmul(rcw.as_dcm(), twp0) + tcw


            # Denote that x-axis is assigned to normal of front plane.
            nvec0_w = rwb0.as_dcm()[:,0]
            depthvec_w = rwc.as_dcm()[:,2]
            degoblique_gt_a = np.arcsin( np.linalg.norm( np.cross(-nvec0_w, depthvec_w) ) )
            degoblique_gt_b = np.arctan2(np.linalg.norm(tcp0[0:1]), tcp0[2] )
            degoblique_gt = max(degoblique_gt_a, degoblique_gt_b)
            degoblique_gt = np.rad2deg(degoblique_gt)

            z_gt = tcp0[2]
            gt_properties[gt_id] = {
                    'min_wh_gt':min(b0.scale[1:]),
                    'z_gt':z_gt,
                    'degoblique_gt':degoblique_gt,
                    }


        for pred_id, pred_area in pred_areas.items():
            if pred_id == 0:
                continue
            pred_max_iou2d[pred_id] = 0.
            correspond = arr_nonbg[ arr_nonbg['pred']==pred_id ]
            occupies = []
            for gt_id, intersection in zip(correspond['gt'], correspond['area']):
                gt_area = gt_areas[gt_id]
                iou2d       = float(intersection) / float(gt_area + pred_area - intersection)
                recall2d    = float(intersection) / float(gt_area)
                precision2d = float(intersection) / float(pred_area)
                pred_max_iou2d[pred_id] = max(pred_max_iou2d[pred_id], iou2d)
                if recall2d > .5:
                    occupies.append( (gt_id, iou2d, recall2d, precision2d) )
            if len(occupies) > 1:
                for gt_id, iou2d, recall2d, precision2d in occupies:
                    pairs[(gt_id, pred_id)] = {'state':'underseg',
                            'iou2d':iou2d, 'recall2d':recall2d, 'precision2d':precision2d }
                    gt_states[gt_id].add('underseg')

        for each in arr_nonbg:
            pred_id, gt_id = each['pred'], each['gt']
            if (gt_id,pred_id) in pairs:
                continue
            intersection       = float(each['area'])
            gt_area, pred_area = gt_areas[gt_id], pred_areas[pred_id]
            iou2d       = intersection / float(gt_area + pred_area - intersection)
            recall2d    = intersection / float(gt_area)
            precision2d = intersection / float(pred_area)
            pairs[(gt_id, pred_id)] \
                    = {'state':'', 'iou2d':iou2d, 'recall2d':recall2d, 'precision2d':precision2d }
        self.profiles = {'pairs':pairs, 'gt_states':gt_states,
                'gt_properties':gt_properties,
                'gt_max_iou2d':gt_max_iou2d, 'pred_max_iou2d':pred_max_iou2d,
                'fp_list':fp_list
                }

        # Visualization
        gt_centers = GetMarkerCenters(gt_marker)
        pred_centers = GetMarkerCenters(pred_marker)
        dst = GetColoredLabel(gt_marker)
        _,contours,_ = cv2.findContours( (pred_marker>0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours) ):
            cv2.drawContours(dst, contours, i, (255,255,255), 2)

        for pred_id, pred_cp in pred_centers.items():
            if not pred_id in fp_list:
                continue
            cv2.putText(dst, "FP %d"%pred_id, pred_cp, cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (150,150,150), 1)

        for gt_id, gt_cp in gt_centers.items():
            msg, states = '', gt_states[gt_id]
            if 'overseg' in states:
                msg += 'o'
            if 'underseg' in states:
                msg += 'u'
            cv2.putText(dst, "%d %s" % (gt_id, msg), gt_cp, cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (10,10,10), 1)
        #cv2.imshow("dst", dst)
        #cv2.waitKey()
        return dst

    def GetMatches(self, obj_array1):
        gt_obbs = self.scene_eval.gt_obbs
        center_poses0, obj_array0 = VisualizeGt(gt_obbs)
        correspondence = GetCorrespondenceMarker()
        infos = MarkerArray()
        marker_optmized_gt = MarkerArray()
        marker_converted_pred = MarkerArray()

        gt_max_iou3d = {}
        markers0 = {}
        for marker in obj_array0.markers:
            gt_max_iou3d[marker.id] = (-1, 0. )
            markers0[marker.id] = marker

        markers1 = {}
        for marker in obj_array1.markers:
            markers1[marker.id] = marker



        for (gt_id,pred_id), pair_info in self.profiles['pairs'].items():
            if pred_id < 0:
                continue
            if not pred_id in markers1: # segment2d exist, but no OBB for it
                continue
            obj0 = markers0[gt_id]
            obj1 = markers1[pred_id]
            b0 = marker2box(obj0)
            b1 = marker2box(obj1)
            b1 = FitAxis(b0, b1)
            rwb0 = rotation_util.from_dcm(b0.rotation)
            rwb1 = rotation_util.from_dcm(b1.rotation)

            daxis = 0
            cp_surf0, _ = GetSurfCenterPoint(obj0,daxis)
            cp_surf1, _ = GetSurfCenterPoint(obj1,daxis)

            dr = rwb1.inv()* rwb0
            deg_err = np.rad2deg( np.linalg.norm(dr.as_rotvec()) )

            # surf_cp is twp for cente 'p'ointr of front plane on 'w'orld frame.
            t_err = cp_surf1 - cp_surf0
            t_err = np.linalg.norm(t_err)

            # Denote that x-axis is assigned to normal of front plane.
            s_err = (b1.scale-b0.scale)[1:]
            s_err = np.abs(s_err)
            max_wh_err = max( s_err ) # [meter]

            # TODO compute daxis
            b0 = GetDepthOptimizedBox(b0, b1, daxis)
            loss = iou.IoU(b0, b1)
            try:
                vol_intersection = loss.intersection()
                iou_val = loss.iou()
            except:
                vol_intersection = 0.
                iou_val = 0.

            precision = vol_intersection /b1.volume
            recall = vol_intersection / b0.volume

            if iou_val > gt_max_iou3d[gt_id][1]:
                gt_max_iou3d[gt_id] = (pred_id, loss.iou() )

            pair_info['iou3d'] = iou_val
            pair_info['precision'] = precision
            pair_info['recall'] = recall
            pair_info['b0'] = b0
            pair_info['b1'] = b1
            pair_info['surf0'] = (cp_surf0,rwb0)
            pair_info['surf1'] = (cp_surf1,rwb1)
            pair_info['t_err']   = t_err
            pair_info['deg_err'] = deg_err
            pair_info['max_wh_err'] = max_wh_err

            info = GetInfoMarker(obj1, precision, recall, iou_val, len(infos.markers) )
            infos.markers.append(info)

            ogt_marker = GetBoxMarker(b0, info.id)
            marker_optmized_gt.markers.append(ogt_marker)

            pred_marker = GetBoxMarker(b1, info.id)
            marker_converted_pred.markers.append(pred_marker)

            correspondence.points.append(Point(cp_surf0[0], cp_surf0[1], cp_surf0[2])) # Front center
            correspondence.points.append(Point(cp_surf1[0], cp_surf1[1], cp_surf1[2]))

        self.profiles['gt_max_iou3d'] = gt_max_iou3d
        if hasattr(self, 'pub_infos'):
            self.pub_correspondence.publish(correspondence)

            a = Marker()
            a.action = Marker.DELETEALL

            infos.markers.append(a)
            infos.markers.reverse()
            self.pub_infos.publish(infos)

            marker_optmized_gt.markers.append(a)
            marker_optmized_gt.markers.reverse()
            self.pub_marker_optmized_gt.publish(marker_optmized_gt)

            marker_converted_pred.markers.append(a)
            marker_converted_pred.markers.reverse()
            self.pub_marker_converted_pred.publish(marker_converted_pred)

        return

    def GetErrors(self, min_iou):
        trans_errors, deg_errors = GetErrors(self.indices0, self.pairs0to1, self.pair_infos, min_iou)
        return trans_errors, deg_errors

def box2marker(obj_box):
    marker = Marker()
    marker.type = Marker.CUBE
    marker.pose.position.x = obj_box.translation[0]
    marker.pose.position.y = obj_box.translation[1]
    marker.pose.position.z = obj_box.translation[2]
    marker.scale.x         = obj_box.scale[0]
    marker.scale.y         = obj_box.scale[1]
    marker.scale.z         = obj_box.scale[2]
    quat = rotation_util.from_dcm(obj_box.rotation).as_quat()
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    return marker

def marker2box(marker):
    quat = marker.pose.orientation
    rot = rotation_util.from_quat([quat.x, quat.y, quat.z, quat.w])
    Rwb = rot.as_dcm()
    whd = np.array( [marker.scale.x, marker.scale.y, marker.scale.z] )
    twb = np.array( [marker.pose.position.x,
                     marker.pose.position.y,
                     marker.pose.position.z])
    w = box.Box()
    b = w.from_transformation( rot.as_dcm(), twb, whd)
    return b


def Posetuple2Rosmsg(tuplepose):
    pose = Pose()
    pose.position.x    = tuplepose[0]
    pose.position.y    = tuplepose[1]
    pose.position.z    = tuplepose[2]
    pose.orientation.w = tuplepose[3]
    pose.orientation.x = tuplepose[4]
    pose.orientation.y = tuplepose[5]
    pose.orientation.z = tuplepose[6]
    return pose

def GetSurfCenterPoint0(pose, scale, daxis):
    # Transform {w}orld <- {b}ox
    twb = pose.position
    twb = np.array((twb.x,twb.y,twb.z))
    q = pose.orientation
    # rotation_util : qx qy qz qw
    rot = rotation_util.from_quat([q.x, q.y, q.z, q.w])
    Rwb = rot.as_dcm()

    # World좌표계의 depth축(daxis)에 가장 가까운것.
    daxis_on_boxcoord = np.argmax(np.abs(Rwb[daxis,:]))

    surf_offset = np.zeros((3,))
    sign = -np.sign(Rwb[daxis,daxis_on_boxcoord])
    surf_offset[daxis_on_boxcoord] = sign*0.5*scale[daxis_on_boxcoord]
    cp_surf = np.matmul(Rwb,surf_offset) + twb
    return cp_surf, rot

def GetSurfCenterPoint(marker, daxis):
    pose = marker.pose
    scale = (marker.scale.x, marker.scale.y, marker.scale.z)
    return GetSurfCenterPoint0(pose, scale, daxis)

def VisualizeGt(gt_obbs, posename='pose_wb'):
    poses = PoseArray()
    poses.header.frame_id = 'robot'
    markers = MarkerArray()

    for i, obj in gt_obbs.items():
        marker = Marker()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.header.frame_id = poses.header.frame_id
        marker.id = i
        marker.pose.position.x = obj['pose_wb'][0]
        marker.pose.position.y = obj['pose_wb'][1]
        marker.pose.position.z = obj['pose_wb'][2]
        marker.scale.x         = obj['scale'][0]
        marker.scale.y         = obj['scale'][1]
        marker.scale.z         = obj['scale'][2]
        quat = obj['pose_wb'][3:]
        marker.pose.orientation.w = quat[0]
        marker.pose.orientation.x = quat[1]
        marker.pose.orientation.y = quat[2]
        marker.pose.orientation.z = quat[3]
        marker.color.a = .5
        marker.color.r = marker.color.g = marker.color.b = .5
        markers.markers.append(marker)
        poses.poses.append(marker.pose)
    return poses, markers

def DrawIouRecall(all_profiles, ax):
    for target, label in {'gt_max_iou2d':'2D segmentation','gt_max_iou3d':'3D OBB'}.items():
        maxIou_datas = []
        for (i_frame,gt_id), (pred_id, iou2d) in all_profiles[target].items():
            if pred_id > 0:
                maxIou_datas.append(iou2d)
        maxIou_datas = np.array(maxIou_datas)
        maxIou_datas = np.sort(maxIou_datas)[::-1] # Descending order.
        n_instance   = len(maxIou_datas)
        recall_arr = (np.arange(1,n_instance+1,dtype=np.float32) )/float(n_instance)

        iou_recall = np.stack( (maxIou_datas,recall_arr), axis=1 )[::-1]
        iou_recall = np.vstack( (np.array((0.,1.)),
                                 iou_recall,
                                 np.array((iou_recall[-1,0],0.)),
                                 np.array((1.,0.)),
                                 ) )
        #print(iou_recall)
        ax.title.set_text('(min IoU-Recall) of detection')
        ax.set_xlabel('min IoU', fontsize=7)
        ax.set_ylabel('Recall', fontsize=7)
        ax.step(iou_recall[:,0], iou_recall[:,1], '-', label=label, where='pre')
        ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7)

    #ax.set_xlim(0.,1.)
    return 

def DrawFScoreTable(ax, ax2, arr_frames, all_profiles):
    # 1) Count FP
    fp_list = {}
    for frame_id, _fp_list in all_profiles['fp_list'].items():
        base = arr_frames['scene_basename'][frame_id]
        if not base in fp_list:
            fp_list[base] = set()
        for pred_id in _fp_list:
            fp_list[base].add( (frame_id,pred_id) )

    # 2) Count TP
    tp_list = {}
    tp_frame_list = {}
    for key, properties in all_profiles['gt_properties'].items():
        frame_id, gt_id = key
        base = arr_frames['scene_basename'][frame_id]
        if not base in tp_list:
            tp_list[base] = set()
        if not frame_id in tp_frame_list:
            tp_frame_list[frame_id] = set()

        state = all_profiles['gt_states'][key]
        if len(state)==0:
            pred_id, iou_val = all_profiles['gt_max_iou3d'][key]
            if iou_val > tp_iou:
                tp_list[base].add( (frame_id,gt_id,pred_id) )
                tp_frame_list[frame_id].add( (gt_id, pred_id) )

    instance_numbers = {}
    frame_numbers = {}
    fn_numbers = {}
    for frame_id, ninstance in enumerate( arr_frames['ninstance'] ):
        base = arr_frames['scene_basename'][frame_id]
        if not base in fn_numbers:
            fn_numbers[base] = 0
            instance_numbers[base] = ninstance
            frame_numbers[base] = 0
        frame_numbers[base] += 1
        fn_numbers[base] += ninstance - len(tp_frame_list[frame_id])

    # 3) Count FN : n(exist object) - TP
    col_labels = ['n(frame)', 'n(box)/frame', 'tp', 'fn', 'fp']
    row_labels = []
    #table_vals = [[n_tp, n_fn, n_fp]]
    table_vals = []

    base_vals = []
    for base_id, base in enumerate(fp_list.keys()):
        nframe = frame_numbers[base]
        nbox = instance_numbers[base]
        row_vec = [nframe, nbox, len(tp_list[base]), fn_numbers[base], len(fp_list[base]) ]
        row_labels.append('#%d'%(base_id+1) )
        table_vals.append(row_vec)
        base_vals.append([base])
    
    # Draw table
    ax.axis('off')
    ax.axis('tight')
    colWidths = [0.15] * len(col_labels)
    colWidths[0] *= 2.2
    colWidths[1] *= 2.7
    the_table = ax.table(cellText=table_vals,
                         colWidths=colWidths,
                         rowLabels=row_labels,
                         colLabels=col_labels,
                         loc='upper right')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7)

    row_labels = [ '#%d'%(i+1) for i in range(len(base_vals) ) ]
    ax2.axis('off')
    ax2.axis('tight')
    the_table = ax2.table(cellText=base_vals,
                         colLabels=['scene name'],
                         colWidths = [.5],
                         rowLabels=row_labels,
                         cellLoc='center',
                         loc='upper right')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7)
    return

def onclick(event):
    global pick_id
    pick_id = event.artist.myidx
    return

def onpress(event):
    global keyevent
    if event.name == 'key_press_event':
        keyevent = event
    return

def GetNeighbors(marker, centers):
    radius = 5
    neighbors = _GetNeighbors(marker, radius=5)
    if False: # verbose
        print(neighbors)
        dst = GetColoredLabel(marker)
        for marker_id, cp in centers.items():
            for nid in neighbors[marker_id]:
                ncp = centers[nid]
                cv2.line(dst, (cp[0],cp[1]), (ncp[0],ncp[1]), (255,255,255), 2)
        for marker_id, cp in centers.items():
            text, font, font_scale, font_thickness \
                    = "%d" % marker_id, cv2.FONT_HERSHEY_SIMPLEX, 1., 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            ox, oy = text_size[0]/2., text_size[1]/2.
            cp = (int(cp[0]-ox), int(cp[1]+oy))
            cv2.rectangle(dst,(cp[0],cp[1]-text_size[1]), (cp[0]+text_size[0],cp[1]), (200,200,200), -1)
            cv2.putText(dst, text, (cp[0],cp[1]), font, font_scale, (0), font_thickness)
        cv2.imshow("dst", dst)
        c = cv2.waitKey(0)
    return neighbors

def GetObliqueError(obb0, obb1):
    x,y,z, qw, qx, qy, qz = obb0['pose_wb']
    rot0 = rotation_util.from_quat([qx, qy, qz, qw])
    x,y,z, qw, qx, qy, qz = obb1['pose_wb']
    rot1 = rotation_util.from_quat([qx, qy, qz, qw])
    R0 = rot0.as_dcm()
    R1 = rot1.as_dcm()
    min_deg_err = 99999.
    #if obb0['id']==7 and obb1['id']==13:
    #    import pdb; pdb.set_trace()
    for i in range(3):
        dotprod = np.abs(np.dot(R0[:,0], R1[:,i]))
        deg_err = np.rad2deg( np.arccos(dotprod) )
        min_deg_err = min(deg_err, min_deg_err)
    return min_deg_err

def DrawApChart(gt_files, segments, arr_frames, all_profiles):
    global pick_id
    global keyevent

    pkg_dir = get_pkg_dir()
    fig = plt.figure(figsize=(16, 16), dpi=100)
    fig.canvas.mpl_connect('key_press_event', onpress)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)
    ax1.set_xlabel('min oblique error among neighbors [deg]')
    ax1.set_ylabel('maximum width among neighbors [pixel]')
    ax2.axis('off')
    ax2.axis('equal')
    ax3.axis('off')
    ax3.axis('equal')

    n_tp = {}
    n_instance = {}
    all_properties = {}

    for key, properties in all_profiles['gt_properties'].items():
        frame_id, gt_id = key
        base = arr_frames['scene_basename'][frame_id]
        all_properties[(base,gt_id)] = properties
        state = all_profiles['gt_states'][key]
        #_, iou_val = all_profiles['gt_max_iou3d'][key]
        _,iou_val = all_profiles['gt_max_iou2d'][key]
        if not (base,gt_id) in n_instance:
            n_instance[(base,gt_id)] = 0
            n_tp[(base,gt_id)] = 0
        n_instance[(base,gt_id)] += 1
        if len(state)==0:
            if iou_val > tp_iou:
                n_tp[(base,gt_id)] +=1
            continue
        if 'overseg' in state:
            pass
        if 'underseg' in state:
            pass

    max_distances = {}
    distances_among_group = {}
    relative_obliqueness = {}
    for base, pick in gt_files.items():
        outline, marker, front_marker, convex_edges\
                = pick['outline'], pick['marker'], pick['front_marker'], pick['convex_edges']
        #edges = np.logical_or(outline, convex_edges)
        #edges[:,0] = edges[:,-1] = edges[0,:] = edges[-1,:] = True
        #cv2.imshow("edges", 255*all_edges.astype(np.uint8))
        #cv2.waitKey()
        obbs, centers = {}, {}
        for obb in pick['obbs']:
            marker_id = obb['id']
            part = front_marker==marker_id
            part[:,0] = part[:,-1] = part[0,:] = part[-1,:] = False
            dist_part = cv2.distanceTransform( part.astype(np.uint8),
                distanceType=cv2.DIST_L2, maskSize=5)
            loc = np.unravel_index( np.argmax(dist_part,axis=None), marker.shape)
            centers[marker_id] = (loc[1],loc[0])
            max_distances[(base,marker_id)] = dist_part.max()
            whd = obb['scale']
            #max_distances[(base,marker_id)] = min(whd[0],whd[1])
            obbs[marker_id] = obb
        
        neighbors = GetNeighbors(marker, centers)
        for id0, obb0 in obbs.items():
            min_d = max_distances[(base,id0)]
            for id1 in neighbors[id0]:
                if max_distances[(base,id1)] < min_d:
                    min_d = max_distances[(base,id1)]
            distances_among_group[(base,id0)] = min_d
            min_oblique = 9999999.
            for id1 in neighbors[id0]:
                obb1 = obbs[id1]
                err = GetObliqueError(obb0, obb1)
                if err < min_oblique:
                    min_oblique = err
            relative_obliqueness[(base,id0)] = min_oblique

    keys, datas = [], np.zeros((len(all_properties), 3))
    for i, (key,properties) in enumerate( all_properties.items() ):
        keys.append(key)
        degoblique_gt, min_wh_gt, z_gt =\
                properties['degoblique_gt'], properties['min_wh_gt'], properties['z_gt']
        if degoblique_gt > 30.:
            continue
        if z_gt > 3.:
            continue
        #max_dist = max_distances[key]
        max_dist = distances_among_group[key]
        max_oblique = relative_obliqueness[key]
        datas[i,0] = max_oblique 
        datas[i,1] = max_dist
        ap = float(n_tp[key]) / float(n_instance[key])
        datas[i,2] = ap
        if ap > .7:
            marker = '.'
        elif ap > .3:
            marker = '^'
        else:
            marker = 'x'
        artist = ax1.scatter(datas[i,0], datas[i,1], marker=marker, picker=True, pickradius=5)
        artist.myidx = i
        fig.canvas.mpl_connect('pick_event', onclick)
    #ax1.set_xlim(0, 50)
    #ax1.set_ylim(0, 100)

    pick_id = None
    while True:
        keyevent, curr_frame = None, -1
        presskey = plt.waitforbuttonpress(.1)
        if keyevent is not None:
            if keyevent.key == 'q':
                #import pdb; pdb.set_trace()
                exit(1)
        if pick_id is None:
            continue
        scene_id, gt_obj_id = keys[pick_id]
        pick = gt_files[scene_id]
        rgb, gt_marker = pick['rgb'], pick['marker']

        cvgt_fn = osp.join(pkg_dir,pick['cvgt_fn'])
        cvgt = cv2.imread(cvgt_fn)
        dst = DrawOutline(cvgt, rgb)
        height, width = cvgt.shape[:2]
        mask = rgb.copy()
        mask[gt_marker!=gt_obj_id,:] = 0
        alpha=.8
        dst = cv2.addWeighted(mask, alpha, dst, 1 - alpha, 0)

        ax2.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), extent = [0,1,0,1],
                aspect=float(height)/float(width) )

        pick_id = None
        while True:
            if pick_id is not None:
                break
            presskey = plt.waitforbuttonpress(.1)
            if presskey is None: # No interaction
                pass
            elif presskey:
                n = len(segments[scene_id])
                if keyevent is not None:
                    if keyevent.key == 'n':
                        curr_frame = (curr_frame+1)%n
                    elif keyevent.key == 'p':
                        curr_frame = (curr_frame+n-1)%n
                    elif keyevent.key == 'q':
                        #import pdb; pdb.set_trace()
                        exit(1)
                    else:
                        keyevent = None
                        continue
                    keyevent = None
                else:
                    keyevent = None
                    continue
            else: # mouse click
                keyevent = None
                break
            if curr_frame < 0:
                curr_frame = 0
            #print('curr_frame = ',curr_frame)
            fn = segments[scene_id][curr_frame]
            dst = cv2.imread(fn)
            ax3.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), extent = [0,1,0,1],
                    aspect=float(height)/float(width) )

    return

def DrawOverUnderHistogram(ax, num_bins, min_max, arr_frames, all_profiles, prop_name, xlabel):
    all_instaces = []
    base_instances = {}
    unique_instances = []
    tp_segmentations = []
    over_segmentations = []
    under_segmentations = []
    for key, properties in all_profiles['gt_properties'].items():
        frame_id, gt_id = key
        base = arr_frames['scene_basename'][frame_id]

        prop = properties[prop_name]
        base_instances[(base, gt_id)] = prop
        all_instaces.append(prop)
        state = all_profiles['gt_states'][key]
        if len(state)==0:
            #_, iou_val = all_profiles['gt_max_iou2d'][key]
            _, iou_val = all_profiles['gt_max_iou3d'][key]
            if iou_val > tp_iou:
                tp_segmentations.append(prop)
            continue
        if 'overseg' in state:
            over_segmentations.append(prop)
        if 'underseg' in state:
            under_segmentations.append(prop)

    for (base, gt_id), prop in base_instances.items():
        unique_instances.append(prop)

    min_max = list(min_max)
    if min_max[0] is None:
        min_max[0] = min(unique_instances)
    if min_max[1] is None:
        min_max[1] = max(unique_instances)
    ntry_hist, bound = np.histogram(all_instaces, num_bins, range=min_max)
    no_samples = ntry_hist==0

    tp_hist    = np.histogram(tp_segmentations, num_bins, range=min_max)[0].astype(np.float)
    over_hist  = np.histogram(over_segmentations, num_bins, range=min_max)[0].astype(np.float)
    under_hist = np.histogram(under_segmentations, num_bins, range=min_max)[0].astype(np.float)
    nbox_hist  = np.histogram(unique_instances, num_bins, range=min_max)[0].astype(np.float)

    x = np.arange(num_bins)
    ntry_hist[no_samples] = 1 # To prevent divide by zero
    if over_hist.max() > .1:
        ax.bar(x-.2, width=.1, height=(over_hist/ntry_hist.astype(np.float))*100.,  alpha=.5, label='oversegment')
    if under_hist.max() > .1:
        ax.bar(x-.1, width=.1, height=(under_hist/ntry_hist.astype(np.float))*100., alpha=.5, label='undersegment')
    if tp_hist.max() > .1:
        ax.bar(x, width=.1, height=(tp_hist/ntry_hist.astype(np.float))*100., alpha=.5, label='3D IoU>%.2f'%tp_iou)
    #ax.bar(x+.1, width=.1, height=(other_hist/ntry_hist.astype(np.float))*100., alpha=.5, label='others')

    xlabels = []
    ntry_hist[no_samples] = 0 # To show true number
    for i in range(num_bins):
        msg = '%.2f~%.2f'%(bound[i],bound[i+1])
        msg += '\nn(box)=%d'%nbox_hist[i]
        msg += '\nn(try)=%d'%ntry_hist[i]
        xlabels.append(msg)

    ax.set_ylabel('[%]',rotation=0, fontsize=7, fontweight='bold')
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=7)
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.yaxis.set_label_coords(-0.08, 1.)
    ax.set_xticks(x)
    ax.set_xlabel(xlabel, fontsize=7, fontweight='bold')
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.18), ncol=3,fontsize=7)


if __name__ == '__main__':
    # For debug
    f = open('/home/geo/catkin_ws/src/ros_unet/tmp.pick','rb')
    pick = pickle.load(f)
    f.close()
    evaluator = Evaluator()
    evaluator.Evaluate(pick['arr_frames'], pick['arr'], is_final=True )

