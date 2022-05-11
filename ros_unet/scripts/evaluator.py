#!/usr/bin/python2
#-*- coding:utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import *
from geometry_msgs.msg import Point, Quaternion, Vector3
from geometry_msgs.msg import PoseArray
import cv2

import pickle
from Objectron import box, iou
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as rotation_util

from unet.gen_obblabeling import ParseGroundTruth

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
    info.text = "precision=%.2f"%precision
    info.text += "\nrecall=%.2f"%recall
    info.text += "\nIoU=%.2f"%loss.iou()
    info.header.frame_id = "robot"
    info.type = Marker.TEXT_VIEW_FACING
    info.scale.z = 0.04
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

def CountUndersegmentation(indices1, pairs1to0, pair_infos, min_recall):
    n_underseg = 0
    for i1 in indices1:
        n = 0
        for i0 in pairs1to0[i1]:
            recall = pair_infos[(i0,i1)]['recall']
            if recall > .5:
                n += 1
        if n > 1:
            n_underseg += 1
    return n_underseg

def CountOversegmentation(indices0, pairs0to1, pair_infos, min_prec):
    n_overseg = 0
    for i0 in indices0:
        n = 0
        for i1 in pairs0to1[i0]:
            precision = pair_infos[(i0,i1)]['precision']
            if precision > min_prec:
                n += 1
        if n > 1:
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
                if deg_err > 45.:
                    import pdb; pdb.set_trace()
                deg_errors.append(deg_err)
                break
    return trans_errors, deg_errors


class Evaluator:
    def __init__(self, pick, Twc, cam_id, verbose=True):
        self.pick = pick
        gt_obbs = pick['obbs']
        # Convert OBB to world(frame_id='robot) coordinate.
        q,t = Twc.orientation, Twc.position
        Rwc = rotation_util.from_quat([q.x, q.y, q.z, q.w])
        twc = np.array((t.x,t.y,t.z))
        for i, obj in enumerate(gt_obbs):
            # Tcb -> Trc * Tcb
            pose_cb = obj['pose']
            Rcb = rotation_util.from_quat([pose_cb[4], pose_cb[5], pose_cb[6], pose_cb[3] ])
            tcb = np.array( pose_cb[:3] ).reshape((3,))
            Rwb = Rwc*Rcb
            twb = np.matmul(Rwc.as_dcm(),tcb) + twc
            q_xyzw = Rwb.as_quat()
            pose_wb = (twb[0], twb[1], twb[2], q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
            obj['pose_wb'] = pose_wb

        # Build KDTree for center position
        centers = np.zeros( (len(gt_obbs),3) )
        for i, gt_obb in enumerate(gt_obbs):
            xyz_qwxyz = gt_obb['pose_wb']
            centers[i,:] = np.array(xyz_qwxyz[:3]).reshape((1,3))
        self.tree = KDTree(centers)
        
        if verbose:
            self.pub_gt_obb = rospy.Publisher("~%s/gt_obb"%cam_id, MarkerArray, queue_size=1)
            self.pub_gt_pose = rospy.Publisher("~%s/gt_pose"%cam_id, PoseArray, queue_size=1)
            self.pub_infos = rospy.Publisher("~%s/info"%cam_id, MarkerArray, queue_size=1)
            self.pub_correspondence = rospy.Publisher("~%s/correspondence"%cam_id, Marker, queue_size=1)
            self.pub_marker_optmized_gt = rospy.Publisher("~%s/optimized_gt"%cam_id, MarkerArray, queue_size=1)
            self.pub_marker_converted_pred = rospy.Publisher("~%s/marker_converted_pred"%cam_id, MarkerArray, queue_size=1)

    def GetMatches(self, obj_array1):
        radius = 1. # serach radius[meter] for center of obj.
        # daxis : World좌표계 축 중, depth 축에 가장 가까운것. 0,1,2<-x,y,z
        #daxis = rospy.get_param("~daxis") #TODO
        daxis = 0

        gt_obbs = self.pick['obbs']
        center_poses0, obj_array0 = VisualizeGt(gt_obbs)
        correspondence = GetCorrespondenceMarker()
        infos = MarkerArray()
        marker_optmized_gt = MarkerArray()
        marker_converted_pred = MarkerArray()

        pairs0to1, pairs1to0, pair_infos = {}, {}, {}
        indices1 = set()
        for i1, obj1 in enumerate(obj_array1.markers):
            if obj1.action == Marker.DELETE:
                continue
            indices1.add(i1)
            pairs1to0[i1] = []

        indices0 = set()
        for i0 in range(len(obj_array0.markers)):
            indices0.add(i0)
            pairs0to1[i0] = []

        for i1, obj1 in enumerate(obj_array1.markers):
            if obj1.action == Marker.DELETE:
                continue
            xyz1 = np.array( [obj1.pose.position.x,
                             obj1.pose.position.y,
                             obj1.pose.position.z])
            b1 = marker2box(obj1)
            candidates = self.tree.query_ball_point(xyz1, radius)

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

                info = GetInfoMarker(obj1, precision, recall, loss, len(infos.markers) )
                infos.markers.append(info)

                ogt_marker = GetBoxMarker(b0, info.id)
                marker_optmized_gt.markers.append(ogt_marker)

                pred_marker = GetBoxMarker(b1, info.id)
                marker_converted_pred.markers.append(pred_marker)

                correspondence.points.append(Point(cp_surf0[0], cp_surf0[1], cp_surf0[2])) # Front center
                correspondence.points.append(Point(cp_surf1[0], cp_surf1[1], cp_surf1[2]))
   
        if hasattr(self, 'pub_infos'):
            self.pub_gt_pose.publish(center_poses0)
            self.pub_gt_obb.publish(obj_array0)

            self.pub_infos.publish(infos)
            self.pub_correspondence.publish(correspondence)
            self.pub_marker_optmized_gt.publish(marker_optmized_gt)
            self.pub_marker_converted_pred.publish(marker_converted_pred)

        return (indices0, indices1, pairs0to1, pairs1to0, pair_infos)


    def put(self, obj_array1):
        # indices0 for ground truth instances
        # indices1 for prediction instances
        indices0, indices1, pairs0to1, pairs1to0, pair_infos = self.GetMatches(obj_array1)
        '''
        TODO
        * [x] Detection recall for each min IoU
        * [x] Detection precision for each min IoU

        * [ ] Mean, std error of center points and rotation of front surface for min IoU

        * [x]  Under segmentation / number of gt instance
            * 한 obb prediction이 둘이상 gt obb에 대해 recall > .5 인 경우.
        * [x] Over segmentation  / number of gt instance
            * 한 gt obb가 둘 이상 obb prediction에 대해, precision > .5 인 경우.
        '''
        n0 = len(indices0)
        n1 = len(indices1)

        min_iou = .5
        # Count detection over min_iou
        n_detection = CountDetection(indices0, pairs0to1, pair_infos, min_iou)
        trans_errors, deg_errors = GetErrors(indices0, pairs0to1, pair_infos, min_iou)

        # ratio_recall    = float(n_detection)/float(n0)
        # ratio_precision = float(n_detection)/float(n1)
        # t_mean, t_std = np.mean(trans_errors), np.std(trans_errors)
        # deg_mean, deg_std = np.mean(deg_errors), np.std(deg_errors)

        # Count undersegmentation
        th = .5
        n_underseg = CountUndersegmentation(indices1, pairs1to0, pair_infos, min_recall=th)
        n_overseg  = CountOversegmentation(indices0, pairs0to1, pair_infos, min_prec=th)

        return
        


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


def GetSurfCenterPoint(marker, daxis):
    # Transform {w}orld <- {b}ox 
    twb = marker.pose.position
    twb = np.array((twb.x,twb.y,twb.z))
    q =marker.pose.orientation
    # rotation_util : qx qy qz qw
    rot = rotation_util.from_quat([q.x, q.y, q.z, q.w])
    Rwb = rot.as_dcm()
    
    # World좌표계의 depth축(daxis)에 가장 가까운것.
    daxis_on_boxcoord = np.argmax(np.abs(Rwb[daxis,:]))

    surf_offset = np.zeros((3,))
    sign = -np.sign(Rwb[daxis,daxis_on_boxcoord])
    scale = (marker.scale.x, marker.scale.y, marker.scale.z)
    surf_offset[daxis_on_boxcoord] = sign*0.5*scale[daxis_on_boxcoord]
    cp_surf = np.matmul(Rwb,surf_offset) + twb
    return cp_surf, rot

def VisualizeGt(gt_obbs):
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

