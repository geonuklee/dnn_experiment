#!/usr/bin/python3
#-*- coding:utf-8 -*-

'''
Compute OBB의 대조군
'''
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose
from scipy.spatial.transform import Rotation as rotation_util
from unet.util import colors

import numpy as np
import pyransac3d as pyrsc # pip3 install pyransac3d

import ros_unet.srv

class Node:
    def __init__(self):
        self.pub_obb = rospy.Publisher("~obbs", MarkerArray, queue_size=1)
        pass

    def ComputeRansacObb2(self, req):
        # TODO 차라리 inner points, outline points 따로받아서,
        # outline points를 깊이방향으로 복사한다음, 두번째 orientation을 구하자.
        pass

    def ComputeRansacObb(self, req):
        #try:
        #    res = self._ComputeRansacObb(req)
        #except:
        #    import pdb; pdb.set_trace()
        #    res = self._ComputeRansacObb(req)
        res = self._ComputeRansacObb(req)
        return res

    def _ComputeRansacObb(self, req):
        thresh = 0.01
        labels = np.array(req.l_clouds)
        xyz_all = np.array(req.xyz_clouds).reshape((labels.shape[0],-1))

        a0 = Marker()
        a0.action = Marker.DELETEALL

        vis_obbs = MarkerArray()
        vis_obbs.markers.append(a0)
        output   = MarkerArray()
        for l in np.unique(labels):
            points = xyz_all[labels==l, :]
            if len(points) < 10:
                continue
            cuboid = pyrsc.Cuboid()
            best_eq, best_inliers = cuboid.fit(points, thresh)
            if len(best_eq) == 0:
                continue

            # x축 기준, on plane 을 찾고, |x|< th points의 yz bounding box로 측정.
            Rbw = best_eq[:,:3] # box <- world
            in_points = points[best_inliers,:]
            each_inliers = []
            n_inliers = []
            in_points1 = np.hstack( (in_points, np.repeat(1.,len(in_points)).reshape(-1,1)) )
            for k in range(3):
                errs = best_eq[k,:].dot(in_points1.T)
                inlier = errs<thresh
                each_inliers.append( inlier )
                n_inliers.append( inlier.sum() )

            front_k = np.argmax(n_inliers)
            front_indices = each_inliers[front_k]
            front_points = in_points[front_indices]
            Xb = np.matmul(Rbw, front_points.T).T
            bmax = np.max(Xb,axis=0)
            bmin = np.min(Xb,axis=0)
            scale = bmax-bmin # TODO depth는 나중에
            cp_b   = .5*(bmax+bmin)
            cp_w = np.matmul( Rbw.T, cp_b.reshape(3,1) ).reshape(-1)

            #Tbw = np.hstack( (Rbw.T, cp.reshape(3,1)) )
            quat_wb = rotation_util.from_dcm(Rbw.T).as_quat()
            marker = Marker()
            marker.type = Marker.CUBE
            marker.header.frame_id = req.frame_id
            color = colors[l % len(colors)]
            marker.color.a = .8
            marker.color.r = float(color[0])/255.
            marker.color.g = float(color[1])/255.
            marker.color.b = float(color[2])/255.
            marker.id = l
            marker.pose.position.x = cp_w[0]
            marker.pose.position.y = cp_w[1]
            marker.pose.position.z = cp_w[2]
            marker.scale.x         = scale[0]
            marker.scale.y         = scale[1]
            marker.scale.z         = scale[2]
            marker.pose.orientation.x = quat_wb[0]
            marker.pose.orientation.y = quat_wb[1]
            marker.pose.orientation.z = quat_wb[2]
            marker.pose.orientation.w = quat_wb[3]
            output.markers.append(marker)
            vis_obbs.markers.append(marker)
        self.pub_obb.publish(vis_obbs)
        output_msg = ros_unet.srv.ComputePoints2ObbResponse()
        output_msg.output = output
        return output_msg

if __name__ == '__main__':
    rospy.init_node('ros_ransacobb_server', anonymous=True)
    node = Node()
    s = rospy.Service('~ComputeObb', ros_unet.srv.ComputePoints2Obb, node.ComputeRansacObb)
    rospy.spin()

