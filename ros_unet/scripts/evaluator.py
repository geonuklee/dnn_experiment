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

from unet.gen_obblabeling import ParseGroundTruth
from os import path as osp

import matplotlib.pyplot as plt
from tabulate import tabulate

th_seg = .3
def IsOversegmentation(precision, recall):
    return (precision-recall) > th_seg

def IsUndersegmentation(precision, recall):
    return (recall-precision) > th_seg

def FitAxis(b0, b1):
    rwb0 = rotation_util.from_dcm(b0.rotation)
    rwb1 = rotation_util.from_dcm(b1.rotation)
    Rwb1 = np.zeros_like(b1.rotation)
    axis_index = [2,2,2]
    for k0 in range(2): # Fit x, y axis.
        axis0 = b0.rotation[:,k0]
        max_abscos, optimal_cos, optimal_k1 = 0., 0.,  -1
        for k1 in range(3):
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
    return b

def GetBest_i1(frame, i0):
    best_i1, max_iou = -1, 0.
    for i1 in frame.pairs0to1[i0]:
        iou = frame.pair_infos[(i0,i1)]['iou']
        if iou > max_iou:
            best_i1, max_iou = i1, iou
    return best_i1, max_iou

def GetBest_i0(frame, i1):
    if i1 < 0:
        return -1, 0.
    best_i0, max_iou = -1, 0.
    for i0 in frame.pairs1to0[i1]:
        iou = frame.pair_infos[(i0,i1)]['iou']
        if iou > max_iou:
            best_i0, max_iou = i0, iou
    return best_i0, max_iou

def GetCorrespondenceMarker():
    correspondence = Marker()
    correspondence.type = Marker.LINE_LIST
    correspondence.header.frame_id = "robot"
    correspondence.pose.orientation.w = 1.
    correspondence.scale.x = 0.01
    correspondence.color.a = 1.
    correspondence.color.r = 1.
    return correspondence

def GetInfoMarker(obj, precision, recall, loss, index):
    info = Marker()
    info.text = "type=%d"%obj.type
    info.text = "pr=%.2f"%precision
    info.text += "\nre=%.2f"%recall
    info.text += "\nIoU=%.2f"%loss.iou()
    info.header.frame_id = "robot"
    info.type = Marker.TEXT_VIEW_FACING
    info.scale.z = 0.02
    if IsOversegmentation(precision,recall):
        info.color.r = 1.
        info.color.g = info.color.b = 0.
        pass
    elif IsUndersegmentation(precision,recall):
        info.color.b = 1.
        info.color.r = info.color.b = 0.
        pass
    else:
        info.color.r = info.color.g = info.color.b = 1.

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

def CountDetection(indices0, pairs0to1, pair_infos, min_iou):
    n_detection = 0
    for i0 in indices0:
        detected = False
        for i1 in pairs0to1[i0]:
            if pair_infos[(i0,i1)]['iou'] > min_iou:
                detected = True
                break
        if detected:
            n_detection += 1
    return n_detection

def CountUndersegmentation(indices1, pairs1to0, pair_infos):
    n_underseg = 0
    for i1 in indices1:
        n = 0
        for i0 in pairs1to0[i1]:
            recall = pair_infos[(i0,i1)]['recall']
            precision = pair_infos[(i0,i1)]['precision']
            if IsUndersegmentation(precision,recall):
                n += 1
        if n > 0:
            n_underseg += 1
    return n_underseg

def CountOversegmentation(indices0, pairs0to1, pair_infos):
    n_overseg = 0
    for i0 in indices0:
        n = 0
        for i1 in pairs0to1[i0]:
            recall = pair_infos[(i0,i1)]['recall']
            precision = pair_infos[(i0,i1)]['precision']
            if IsOversegmentation(precision,recall):
                n += 1
        if n > 0: # 모서리 귀퉁이에 oversegmentation이 겹친경우.
            n_overseg += 1
    return n_overseg

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

    def Evaluate(self, arr_scense=None, arr_frames=None, arr=None, is_final=False):
        self.n_evaluate += 1
        if arr is None:
            arr_scense, arr_frames, arr = self.GetTables()
            if is_final:
                f = open('/home/geo/catkin_ws/src/ros_unet/tmp.pick','wb')
                pickle.dump({
                    'arr_scense':arr_scense,
                    'arr_frames':arr_frames,
                    'arr':arr,
                    }, f)
                f.close()

        '''
        * [x] Accumulative graph : minIoU - Recall ratio
            * ref: https://matplotlib.org/stable/gallery/statistics/histogram_cumulative.html
        * [x] Histogram : min(w,h) - prob(UnderSeg)*, prob(OverSeg) << for skew less than 20deg
        * [x] Histogram : skew angle - prob(UnderSeg), prob(OverSeg)* << for min(w,h) over 10cm
        * [x] Add histogram - 'IoU > .7 ratio for each cases'
        * [x] GetNSample
        * [x] Table : 
        '''
        if not hasattr(self, 'fig'):
            self.fig = plt.figure(figsize=(12, 9))
        else:
            plt.clf()
        fig = self.fig

        sub_rc = (2,2)
        ax = fig.add_subplot(sub_rc[0], sub_rc[1], 1)
        DrawIouRecall(arr,ax)
        tp_iou = .7
        tp = np.logical_and(arr['maxIoU'] > tp_iou, arr['crosscheck'])

        num_bins = 5
        ax = fig.add_subplot(sub_rc[0], sub_rc[1], 2)
        ax.title.set_text('min(w,h)')
        DrawOverUnderHistogram(ax, num_bins, (arr['min_wh_gt'].min() , arr['min_wh_gt'].max() ),
                tp_iou, tp, arr, arr['min_wh_gt'], '[m]')

        ax = fig.add_subplot(sub_rc[0], sub_rc[1], 3)
        ax.title.set_text('center depth')
        DrawOverUnderHistogram(ax, num_bins, (0.5, arr['z_gt'].max()),
                tp_iou, tp, arr, arr['z_gt'], '[m]')

        ax = fig.add_subplot(sub_rc[0], sub_rc[1], 4)
        ax.title.set_text('skew angle')
        DrawOverUnderHistogram(ax, num_bins, (0., 50.),
                tp_iou, tp, arr, arr['degskew_gt'], '[deg]')

        n_instance = arr_scense['n_instance']
        etimes = arr_frames['elapsed_time']
        print('n(Scene) : %d' %  arr_scense.shape[0])
        print('n(Frame) : %d' %  arr_frames.shape[0])
        print('n(Box) : ~N(%.2f,%.2f)' % (np.mean(n_instance), np.std(n_instance)) )

        dtype = [('param',object), ('max',float),('median',float) ]
        values = []
        for param in ['deg_err', 't_err', 'max_wh_err']:
            data = arr[param][tp]
            value = (param, np.max(data), np.median(data) )
            values.append(value)
        values.append( ('elapsed time', np.max(etimes), np.median(etimes) ) )
        arr_values = np.array(values,dtype=dtype)
        tb = tabulate( arr_values, arr_values.dtype.names )
        print(tb)

        if is_final:
            fig.suptitle('Evaluation', fontsize=16)
        plt.tight_layout(pad=3., w_pad=2., h_pad=3.0)
        fig.canvas.draw()
        plt.show(block=is_final) # block if this is final call to check.
        return

    def GetTables(self):
        dtype = [('maxIoU', float),
                 ('recall', float),
                 ('precision', float),
                 ('t_err', float),
                 ('deg_err', float),
                 ('max_wh_err', float),
                 ('min_wh_gt', float),
                 ('z_gt', float),
                 ('degskew_gt', float),
                 ('overseg', bool),
                 ('underseg', bool),
                 ('under(frame,i1)', tuple), # frame and prediction index for unique
                 ('scene_basename', object),
                 ('crosscheck', bool),
                 ('base_id', int),
                 ('frame_id', int),
                 ('i0', int),
                 ('i1', int),
                 ]

        scene_dtype = [ 
                ('scene_basename', object),
                ('rosbag_fn', object),
                ('cvgt_fn', object),
                ('n_instance', int),
                 ]

        frame_dtype = [
                ('scene_basename', object),
                ('elapsed_time',float),
                ]

        scenes = []
        base2id = {}
        for base_id, (base, scen_eval) in enumerate(self.scene_evals.items()):
            base2id[base] = base_id
            pick = scen_eval.pick
            n_instance = len(pick['obbs'])
            scenes.append( (base, pick['fullfn'], pick['cvgt_fn'], n_instance) )

        arr_scenes = np.array(scenes, dtype=scene_dtype)
        frames = []
        frame_evals= []
        for base, frame_evals_of_a_scene in self.frame_evals.items():
            frame_evals += frame_evals_of_a_scene
            pick = frame_evals_of_a_scene[0].scene_eval.pick
            #n_instance = len(pick['obbs'])
            #n_frames = len(frame_evals_of_a_scene)
            for frame_eval in frame_evals_of_a_scene:
                etime = frame_eval.elapsed_time
                frames.append( (base, etime) )
        arr_frames = np.array(frames, dtype=frame_dtype)

        values = []
        for frame in frame_evals:
            pick = frame.scene_eval.pick
            rosbag_fn, cvgt_fn = pick['fullfn'], pick['cvgt_fn']
            scene_basename = osp.splitext(osp.basename(rosbag_fn))[0]

            for i0 in frame.indices0:
                #1) Get Best IoU and w0, h0,
                #2) Get Best IoU's 'w1, h1, b(IsOverseg), b(IsUnderseg), t_err, r_err, Twb'
                best_i1, max_iou  = GetBest_i1(frame,i0)
                correspond_i0, _  = GetBest_i0(frame,best_i1)
                crosscheck = i0 == correspond_i0

                if best_i1 < 0:
                    recall, precision = 0., 0.
                    t_err, deg_err, max_wh_err = 0., 0., 0.
                    min_wh_gt, z_gt, degskew_gt = 0., 0., 0.
                    overseg, underseg, underseg_frame_i1 = False, False, ()
                else:
                    info = frame.pair_infos[(i0,best_i1)]
                    recall    = info['recall']
                    precision = info['precision']
                    b0, b1 = info['b0'], info['b1']
                    b1 = FitAxis(b0, b1)

                    rwb0 = rotation_util.from_dcm(b0.rotation)
                    rwb1 = rotation_util.from_dcm(b1.rotation)
                    dr = rwb1.inv()* rwb0
                    deg_err = np.rad2deg( np.linalg.norm(dr.as_rotvec()) )

                    #if deg_err > 45.:
                    #    import pdb; pdb.set_trace()

                    # surf_cp is twp for cente 'p'ointr of front plane on 'w'orld frame.
                    t_err = info['surf1'][0] - info['surf0'][0]
                    t_err = np.linalg.norm(t_err)

                    # Denote that x-axis is assigned to normal of front plane.
                    s_err = (b1.scale-b0.scale)[1:]
                    s_err = np.abs(s_err)
                    max_wh_err = max( s_err ) # [meter]
                    min_wh_gt = min(b0.scale[1:])
                    # Center 'p'oint of ground truth on 'c'amera frame.
                    (rwc, twc) = frame.scene_eval.Twc
                    twp0 = info['surf0'][0]
                    rcw , tcw = rwc.inv(), -np.matmul(rwc.inv().as_dcm(), twc)
                    tcp0 = np.matmul(rcw.as_dcm(), twp0) + tcw
                    z_gt = tcp0[2]

                    # Denote that x-axis is assigned to normal of front plane.
                    nvec0_w = rwb0.as_dcm()[:,0]
                    depthvec_w = rwc.as_dcm()[:,2]
                    degskew_gt = np.arcsin( np.linalg.norm( np.cross(-nvec0_w, depthvec_w) ) )
                    degskew_gt = np.rad2deg(degskew_gt)

                    overseg = IsOversegmentation( precision, recall)
                    underseg= IsUndersegmentation(precision, recall)
                    if underseg:
                        underseg_frame_i1 = (frame.frame_id, best_i1)
                    else:
                        underseg_frame_i1 = (-1,-1)

                value = ( max_iou, recall, precision,
                        t_err, deg_err, max_wh_err, min_wh_gt, z_gt, degskew_gt,
                        overseg, underseg, underseg_frame_i1,
                        scene_basename,
                        crosscheck,
                        base2id[scene_basename],
                        frame.frame_id,
                        i0, best_i1,
                        )
                values.append(value)
                # for i0 in frame.indices0
            # for frame
        arr = np.array(values, dtype=dtype)
        return arr_scenes, arr_frames, arr

class SceneEval:
    def __init__(self, pick, Twc, plane_c, max_z, cam_id):
        self.pick = pick
        # Convert OBB to world(frame_id='robot) coordinate.
        q,t = Twc.orientation, Twc.position
        Rwc = rotation_util.from_quat([q.x, q.y, q.z, q.w])
        twc = np.array((t.x,t.y,t.z))
        self.Twc = (Rwc, twc)

        gt_obbs = []
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
            gt_obbs.append(obj)
        self.gt_obbs = gt_obbs

        # Build KDTree for center position
        centers = np.zeros( (len(gt_obbs),3) )
        for i, gt_obb in enumerate(gt_obbs):
            xyz_qwxyz = gt_obb['pose_wb']
            centers[i,:] = np.array(xyz_qwxyz[:3]).reshape((1,3))
        self.tree = KDTree(centers)

        self.pub_gt_obb = rospy.Publisher("~%s/gt_obb"%cam_id, MarkerArray, queue_size=1)
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

    def GetMatches(self, obj_array1):
        radius = 1. # serach radius[meter] for center of obj.
        # daxis : World좌표계 축 중, depth 축에 가장 가까운것. 0,1,2<-x,y,z
        #daxis = rospy.get_param("~daxis") #TODO
        daxis = 0
        gt_obbs = self.scene_eval.gt_obbs
        center_poses0, obj_array0 = VisualizeGt(gt_obbs)
        correspondence = GetCorrespondenceMarker()
        infos = MarkerArray()
        marker_optmized_gt = MarkerArray()
        marker_converted_pred = MarkerArray()

        pairs0to1, pairs1to0, pair_infos = {}, {}, {}
        indices1 = set()
        for i1, obj1 in enumerate(obj_array1.markers):
            if obj1.type != Marker.CUBE or obj1.action != Marker.ADD:
                continue
            indices1.add(i1)
            pairs1to0[i1] = []

        indices0 = set()
        for i0 in range(len(obj_array0.markers)):
            indices0.add(i0)
            pairs0to1[i0] = []

        n_iou_over_th, min_iou = 0, .7
        for i1, obj1 in enumerate(obj_array1.markers):
            if obj1.type != Marker.CUBE or obj1.action != Marker.ADD:
                continue
            xyz1 = np.array( [obj1.pose.position.x,
                             obj1.pose.position.y,
                             obj1.pose.position.z])
            b1 = marker2box(obj1)
            candidates = self.scene_eval.tree.query_ball_point(xyz1, radius)

            if len(candidates) == 0:
                continue
            kv = []
            for i0 in candidates:
                b0 = marker2box(obj_array0.markers[i0])
                loss = iou.IoU(b0, b1)
                try :
                    a = loss.intersection()
                except:
                    continue
                if a > 0.:
                    dx = b0.translation-b1.translation
                    kv.append((i0, np.linalg.norm(dx)))
                    #kv.append( (i0, -loss.intersection() ) )

            if len(kv) == 0:
                continue
            kv = sorted(kv, key=lambda x: x[1] , reverse=False)
            #i0 = kv[0][0]
            for rank, (i0,_) in enumerate(kv):
                obj0 = obj_array0.markers[i0]
                b0 = marker2box(obj0)
                loss = iou.IoU(b0, b1)

                vol_intersection = loss.intersection()
                if vol_intersection < 0.05*b1.volume:
                    continue

                b0 = GetDepthOptimizedBox(b0, b1, daxis)
                loss = iou.IoU(b0, b1)
                vol_intersection = loss.intersection()
                precision = vol_intersection /b1.volume
                recall = vol_intersection / b0.volume

                if vol_intersection <= 0.:
                    continue

                cp_surf0, rwb0 = GetSurfCenterPoint(obj0,daxis)
                cp_surf1, rwb1 = GetSurfCenterPoint(obj1,daxis)

                pairs0to1[i0].append(i1)
                pairs1to0[i1].append(i0)

                pair_infos[(i0,i1)] = { 'iou':loss.iou(), 'precision':precision, 'recall':recall,
                        'b0':b0, 'b1':b1, 'surf0':(cp_surf0,rwb0), 'surf1':(cp_surf1,rwb1), }
                if rank != 0:
                    continue

                if precision < 0.1 or recall < 0.1:
                    continue

                if loss.iou() > min_iou:
                    n_iou_over_th += 1

                info = GetInfoMarker(obj1, precision, recall, loss, len(infos.markers) )
                infos.markers.append(info)

                ogt_marker = GetBoxMarker(b0, info.id)
                marker_optmized_gt.markers.append(ogt_marker)

                pred_marker = GetBoxMarker(b1, info.id)
                marker_converted_pred.markers.append(pred_marker)

                correspondence.points.append(Point(cp_surf0[0], cp_surf0[1], cp_surf0[2])) # Front center
                correspondence.points.append(Point(cp_surf1[0], cp_surf1[1], cp_surf1[2]))
        #print("IoU>%.2f for each i0 =  %d/%d" % (min_iou,n_iou_over_th, len(obj_array0.markers) ) )

        if hasattr(self, 'pub_infos'):
            self.pub_correspondence.publish(correspondence)

            a = Marker()
            a.action = Marker.DELETEALL
            self.scene_eval.pubGtObb()

            infos.markers.append(a)
            infos.markers.reverse()
            self.pub_infos.publish(infos)

            marker_optmized_gt.markers.append(a)
            marker_optmized_gt.markers.reverse()
            self.pub_marker_optmized_gt.publish(marker_optmized_gt)

            marker_converted_pred.markers.append(a)
            marker_converted_pred.markers.reverse()
            self.pub_marker_converted_pred.publish(marker_converted_pred)

        self.indices0, self.indices1, self.pairs0to1, self.pairs1to0, self.pair_infos = \
                indices0, indices1, pairs0to1, pairs1to0, pair_infos
        return

    def CountDetection(self, min_iou):
        n_detection = CountDetection(self.indices0, self.pairs0to1, self.pair_infos, self.min_iou)
        return n_detection

    def GetErrors(self, min_iou):
        trans_errors, deg_errors = GetErrors(self.indices0, self.pairs0to1, self.pair_infos, min_iou)
        return trans_errors, deg_errors

    def Depricate(self, min_iou=.5):
        # indices0 for ground truth instances
        # indices1 for prediction instances
        n0 = len(self.indices0)
        n1 = len(self.indices1)

        # Count detection over min_iou
        n_detection = self.CountDetection(min_iou)
        trans_errors, deg_errors = self.GetErrors(self.indices0, self.pairs0to1, self.pair_infos, min_iou)
        # ratio_recall    = float(n_detection)/float(n0)
        # ratio_precision = float(n_detection)/float(n1)
        # t_mean, t_std = np.mean(trans_errors), np.std(trans_errors)
        # deg_mean, deg_std = np.mean(deg_errors), np.std(deg_errors)
        return n0, n1, n_detection, trans_errors, deg_errors

    def CountUnderOverSegmentation(self):
        indices0, indices1, pairs0to1, pairs1to0, pair_infos \
                = self.indices0, self.indices1, self.pairs0to1, self.pairs1to0, self.pair_infos
        n0 = len(indices0)
        n1 = len(indices1)
        # Count undersegmentation
        n_underseg = CountUndersegmentation(indices1, pairs1to0, pair_infos)
        n_overseg  = CountOversegmentation(indices0, pairs0to1, pair_infos)
        return n0, n1, n_underseg, n_overseg


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

    for i, obj in enumerate(gt_obbs):
        marker = Marker()
        marker.type = Marker.CUBE
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

def DrawIouRecall(arr, ax):
    # TODO n_instance
    #partial_arr = fields_view(arr, ['frame_id', 'i0', 'maxIoU', 'crosscheck'])
    cases = {}
    for row in range(arr.shape[0]):
        frame_id, i0, maxIoU, crosscheck \
                = arr['frame_id'][row], arr['i0'][row], arr['maxIoU'][row], arr['crosscheck'][row]
        k = (frame_id,i0)
        if cases.has_key(k):
            if maxIoU > cases[k] and crosscheck:
                cases[k] = maxIoU
        else:
            if crosscheck:
                cases[k] = maxIoU
            else:
                cases[k] = 0
    maxIou_datas =  np.array(cases.values())
    maxIou_datas = np.sort(maxIou_datas)[::-1] # Descending order.
    n_instance   = len(cases)
    recall_arr = (np.arange(0,n_instance,dtype=np.float32) + 1.)/float(n_instance)
    iou_recall = np.stack( (maxIou_datas,recall_arr), axis=1 )[::-1]
    iou_recall = iou_recall[iou_recall[:,0]>0.,:]

    ax.title.set_text('(minIoU-Recall) of detection')
    ax.set_xlabel('minIoU')
    ax.set_ylabel('Recall')
    ax.plot(iou_recall[:,0], iou_recall[:,1],'-')
    return 

def DrawOverUnderHistogram(ax, num_bins, min_max, tp_iou, tp, arr, param, xlabel):
    # 4 cases
    # TP : IoU > tp_iou
    # Underseg
    # Overseg
    # Others

    overseg   = arr['overseg']
    undersegk = arr['under(frame,i1)']
    underseg  = arr['underseg']

    #underseg[3] = True
    #underseg[10] = True
    #overseg[0] = True
    #overseg[4] = overseg[7] = True

    indicies = np.arange(underseg.shape[0])[underseg]
    _, unique = np.unique( undersegk[underseg], return_index=True)
    unique_underseg = indicies[unique]

    #union = np.logical_or(overseg, underseg)
    #union = np.logical_or(union, tp)
    #others = ~union

    base_i0 = np.stack((arr['base_id'], arr['i0']), axis=1) # axis=1 cause shape (N,2)
    assert(base_i0.shape[0] == len(arr) )
    _, unique_boxes = np.unique( base_i0, return_index=True, axis=0)
    ntry_hist, bound = np.histogram(param, num_bins, range=min_max)
    nbox_hist, _     = np.histogram(param[unique_boxes], num_bins, range=min_max)

    no_samples = ntry_hist==0

    over_hist  = np.histogram(param[overseg] , num_bins, range=min_max)[0].astype(np.float)
    tp_hist    = np.histogram(param[tp], num_bins, range=min_max)[0].astype(np.float)
    under_hist = np.histogram(param[unique_underseg] , num_bins, range=min_max)[0].astype(np.float)
    #other_hist = np.histogram(param[others], num_bins, range=min_max)[0].astype(np.float)

    x = np.arange(num_bins)
    ntry_hist[no_samples] = 1 # To prevent divide by zero
    ax.bar(x-.2, width=.1, height=(over_hist/ntry_hist.astype(np.float))*100.,  alpha=.5, label='oversegment')
    ax.bar(x-.1, width=.1, height=(under_hist/ntry_hist.astype(np.float))*100., alpha=.5, label='undersegment')
    ax.bar(x, width=.1, height=(tp_hist/ntry_hist.astype(np.float))*100., alpha=.5, label='IoU>%.2f'%tp_iou)
    #ax.bar(x+.1, width=.1, height=(other_hist/ntry_hist.astype(np.float))*100., alpha=.5, label='others')

    xlabels = []
    ntry_hist[no_samples] = 0 # To show true number
    for i in range(num_bins):
        msg = '%.2f~%.2f'%(bound[i],bound[i+1])
        msg += '\nn(box)=%d'%nbox_hist[i]
        msg += '\nn(try)=%d'%ntry_hist[i]
        xlabels.append(msg)

    ax.set_ylabel('[%]',rotation=0, fontweight='bold')
    ax.set_xticklabels(xlabels, rotation=0.,fontsize=6)
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.yaxis.set_label_coords(-0.08, 1.)
    ax.set_xticks(x)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.12), ncol=3)


if __name__ == '__main__':
    # For debug
    f = open('/home/geo/catkin_ws/src/ros_unet/tmp.pick','rb')
    pick = pickle.load(f)
    f.close()
    evaluator = Evaluator()
    evaluator.Evaluate(pick['arr_scense'], pick['arr_frames'], pick['arr'], is_final=True )

