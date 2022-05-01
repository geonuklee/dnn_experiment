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

#from unet.gen_obblabeling 

class Collector:
    def __init__(self):
        self.msg = None

    def callback(self, msg):
        self.msg = msg

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
    return cp_surf

def VisualizeGt(gt_obbs):
    poses = PoseArray()
    poses.header.frame_id = 'robot'
    markers = MarkerArray()

    for i, obj in enumerate(gt_obbs):
        marker = Marker()
        marker.type = Marker.CUBE
        marker.header.frame_id = poses.header.frame_id
        marker.id = i
        marker.pose.position.x = obj['pose'][0]
        marker.pose.position.y = obj['pose'][1]
        marker.pose.position.z = obj['pose'][2]
        marker.scale.x         = obj['scale'][0]
        marker.scale.y         = obj['scale'][1]
        marker.scale.z         = obj['scale'][2]
        quat = obj['pose'][3:]
        marker.pose.orientation.w = quat[0]
        marker.pose.orientation.x = quat[1]
        marker.pose.orientation.y = quat[2]
        marker.pose.orientation.z = quat[3]
        marker.color.a = .5
        marker.color.r = marker.color.g = marker.color.b = .5
        markers.markers.append(marker)
        poses.poses.append(marker.pose)
    return poses, markers

if __name__ == '__main__':
    rospy.init_node('ros_eval', anonymous=True)
    rate = rospy.Rate(hz=100)

    collector = Collector()
    sub_object_array = rospy.Subscriber("~estimation", MarkerArray, collector.callback, queue_size=1);
    pub_gt_obb = rospy.Publisher("~gt_obb", MarkerArray, queue_size=1)
    pub_gt_pose = rospy.Publisher("~gt_pose", PoseArray, queue_size=1)
    pub_infos = rospy.Publisher("~info", MarkerArray, queue_size=1)
    pub_correspondence = rospy.Publisher("~correspondence", Marker, queue_size=1)
    pub_marker_optmized_gt = rospy.Publisher("~optimized_gt", MarkerArray, queue_size=1)
    pub_marker_converted_pred = rospy.Publisher("~marker_converted_pred", MarkerArray, queue_size=1)
    param_name = "/cam0/base_T_cam"
    while not rospy.has_param(param_name):
        rate.sleep()
    Trc = rospy.get_param(param_name)
    Trc = np.array(Trc).reshape((4,4))
    Rrc = rotation_util.from_dcm(Trc[:3,:3])
    trc = Trc[:3,3].reshape((3,))

    # daxis : World좌표계 축 중, depth 축에 가장 가까운것. 0,1,2<-x,y,z
    #daxis = rospy.get_param("~daxis") #TODO
    daxis = 0

    with open('/home/geo/catkin_ws/src/ros_unet/tmp1.pick','rb') as f:
        pick = pickle.load(f)
        gt_obbs = pick['obbs']
        for i, obj in enumerate(gt_obbs):
            # Tcb -> Trc * Tcb
            pose_cb = obj['pose']
            Rcb = rotation_util.from_quat([pose_cb[4], pose_cb[5], pose_cb[6], pose_cb[3] ])
            tcb = np.array( pose_cb[:3] ).reshape((3,))
            Rrb = Rrc*Rcb
            trb = np.matmul(Rrc.as_dcm(),tcb) + trc
            q_xyzw = Rrb.as_quat()
            pose_rb = (trb[0], trb[1], trb[2], q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
            obj['pose'] = pose_rb

    vis_pose, vis_obb = VisualizeGt(gt_obbs)

    n = len(gt_obbs)
    centers = np.zeros( (n,3) )

    for i, gt_obb in enumerate(gt_obbs):
        xyz_qwxyz = gt_obb['pose']
        centers[i,:] = np.array(xyz_qwxyz[:3]).reshape((1,3))

    tree = KDTree(centers)
    radius = 1. # search radius [meter]

    while not rospy.is_shutdown():
        if collector.msg is None:
            rate.sleep()
            continue
        obj_array = collector.msg
        collector.msg = None
        pub_gt_pose.publish(vis_pose)
        pub_gt_obb.publish(vis_obb)

        correspondence = Marker()
        correspondence.type = Marker.LINE_LIST
        correspondence.header.frame_id = "robot"
        correspondence.pose.orientation.w = 1.
        correspondence.scale.x = 0.01
        correspondence.color.a = 1.
        correspondence.color.r = 1.
        infos = MarkerArray()
        marker_optmized_gt = MarkerArray()
        marker_converted_pred = MarkerArray()
        for obj in obj_array.markers:
            if obj.action == Marker.DELETE:
                continue
            xyz1 = np.array( [obj.pose.position.x,
                             obj.pose.position.y,
                             obj.pose.position.z])
            b1 = marker2box(obj)
            candidates = tree.query_ball_point(xyz1, radius)

            if len(candidates) == 0:
                continue
            kv = []
            for j in candidates:
                b0 = marker2box(vis_obb.markers[j])
                loss = iou.IoU(b0, b1)
                try :
                    a = loss.intersection()
                except:
                    continue
                if a > 0.:
                    dx = b0.translation-b1.translation
                    kv.append((j, -np.linalg.norm(dx)))
                    #kv.append( (j, loss.intersection() ) )
            if len(kv) == 0:
                continue

            kv = sorted(kv, key=lambda x: x[1] , reverse=True)
            j = kv[0][0]
            obj0 = vis_obb.markers[j]
            b0 = marker2box(obj0)
            loss = iou.IoU(b0, b1)

            vol_intersection = loss.intersection()
            if vol_intersection < 0.05*b1.volume:
                continue

            Rwb0 = b0.rotation

            # World좌표계의 depth 축에 가장 가까운것.
            optimized_axis = np.argmax(np.abs(Rwb0[daxis,:]))

            # depth axis의 반대방향으로
            sign = -np.sign(Rwb0[daxis,optimized_axis])

            optimized_gt_whd = b0.scale.copy()
            axis1 = np.argmax(np.abs(b1.rotation[daxis,:]))
            optimized_val = min(optimized_gt_whd[optimized_axis], b1.scale[axis1])
            optimized_gt_whd[optimized_axis] = optimized_val

            offset = np.zeros((3,) )
            offset[optimized_axis] = sign*0.5*(b0.scale[optimized_axis]-optimized_val)
            dX = np.matmul(Rwb0,offset).reshape((-1,))
            optimized_gt_xyz = b0.translation + dX
            w0 = box.Box()
            b0 = w0.from_transformation(Rwb0, optimized_gt_xyz, optimized_gt_whd)

            loss = iou.IoU(b0, b1)
            vol_intersection = loss.intersection()
            precision = vol_intersection /b1.volume 
            recall = vol_intersection / b0.volume
            if precision < 0.1 or recall < 0.1:
                continue
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
            info.id = len(infos.markers)
            infos.markers.append(info)

            ogt_marker = box2marker(b0)
            ogt_marker.id = info.id
            ogt_marker.color.a = 0.8
            ogt_marker.color.r = 1.
            ogt_marker.color.g = ogt_marker.color.b = 0.2
            ogt_marker.header.frame_id = "robot"
            marker_optmized_gt.markers.append(ogt_marker)

            pred_marker = box2marker(b1)
            pred_marker.id = info.id
            pred_marker.color.a = 0.8
            pred_marker.color.b = 1.
            pred_marker.color.r = ogt_marker.color.b = 0.2
            pred_marker.header.frame_id = "robot"
            marker_converted_pred.markers.append(pred_marker)

            cp_surf0 = GetSurfCenterPoint(obj0,daxis)
            cp_surf1 = GetSurfCenterPoint(obj, daxis)
            correspondence.points.append(Point(cp_surf0[0], cp_surf0[1], cp_surf0[2])) # Front center
            correspondence.points.append(Point(cp_surf1[0], cp_surf1[1], cp_surf1[2]))

            pub_infos.publish(infos)
            pub_correspondence.publish(correspondence)
            pub_marker_optmized_gt.publish(marker_optmized_gt)
            pub_marker_converted_pred.publish(marker_converted_pred)

            # 정보 저장
            # iou - precision - recall - (gt)w - h - d - min_width - distance - iter - filename - gt_name
            ''' Log below error for evaluation..
            b0.scale
            b0.rotation
            dt = np.linalg.norm(cp_surf1-cp_surf0)
            '''
            msg = "%.3f"%loss.iou()
            msg += ",%.3f"%precision
            msg += ",%.3f"%recall
            msg += ",%.3f"%obj0.scale.x
            msg += ",%.3f"%obj0.scale.y
            msg += ",%.3f"%obj0.scale.z
            msg += "\n"
 
        rate.sleep()
