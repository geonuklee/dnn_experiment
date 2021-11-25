#!/usr/bin/python2
#-*- coding:utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import numpy as np

import tensorflow as tf
from bonet import BoNet, Data_Configs
import os

from helper_data_plot import Plot as Plot

class Sub:
    def __init__(self, topic):
        self.subscriber = rospy.Subscriber(topic, PointCloud2, self.callback, queue_size=1)
        self.bat_pc = None

    def callback(self, cloud):
        if self.bat_pc is not None:
            return

        dtype_list = ros_numpy.point_cloud2.fields_to_dtype(cloud.fields, cloud.point_step)
        array = np.fromstring(cloud.data, dtype_list)
        # array['rgb'] -> rgb['r'], rgb['g'], rgb['b']
        rgb = ros_numpy.point_cloud2.split_rgb_field(array)

        xyz = np.vstack((array['x'],array['y'],array['z'])).transpose()
        rgb = np.vstack((rgb['r'], rgb['g'], rgb['b'])).transpose()
        # TODO normalize
        self.bat_pc = np.hstack((xyz, xyz, rgb, xyz)).reshape((1,-1,12) )

if __name__ == '__main__':
    rospy.init_node('ros_bonet', anonymous=True)
    sub = Sub(topic='~input')
    rate = rospy.Rate(hz=30)

    configs = Data_Configs()
    net = BoNet(configs=configs)
    ####### 1. networks
    net.X_pc = tf.placeholder(shape=[None, None, net.points_cc], dtype=tf.float32, name='X_pc')
    net.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
    with tf.variable_scope('backbone'):
        net.point_features, net.global_features, net.y_psem_pred = net.backbone_pointnet2(net.X_pc, net.is_train)
    with tf.variable_scope('bbox'):
        net.y_bbvert_pred_raw, net.y_bbscore_pred_raw = net.bbox_net(net.global_features)
    with tf.variable_scope('pmask'):
        net.y_pmask_pred_raw = net.pmask_net(net.point_features, net.global_features, net.y_bbvert_pred_raw, net.y_bbscore_pred_raw)

    ####### 2. restore trained model
    model_path ='/home/docker/catkin_ws/src/dnn_experiment/3D-BoNet/model_released/model.cptk'
    if not os.path.isfile(model_path + '.data-00000-of-00001'):
        print(model_path)
        print ('please download the released model!')
        exit(0)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.visible_device_list = '0'
    net.sess = tf.Session(config=config)
    tf.train.Saver().restore(net.sess, model_path)
    print('Model restored sucessful!')

    while not rospy.is_shutdown():
        rate.sleep()
        if sub.bat_pc is None:
            continue
        bat_pc = sub.bat_pc
        print(bat_pc.shape)
        [y_psem_pred_sq_raw, y_bbvert_pred_sq_raw, y_bbscore_pred_sq_raw, y_pmask_pred_sq_raw] = \
            net.sess.run([net.y_psem_pred, net.y_bbvert_pred_raw, net.y_bbscore_pred_raw, net.y_pmask_pred_raw],feed_dict={net.X_pc: bat_pc[:, :, 0:9], net.is_train: False})

        #### if you need to visulize, please uncomment the follow lines
        # class가 뭐지?, Bounding box는 어디?
        #b = 0
        #pc = np.asarray(bat_pc[b], dtype=np.float16)
        #sem_gt = np.asarray(bat_sem_gt[b], dtype=np.int16)
        #ins_gt = np.asarray(bat_ins_gt[b], dtype=np.int32)
        #sem_pred_raw = np.asarray(y_psem_pred_sq_raw[b], dtype=np.float16)
        #bbvert_pred_raw = np.asarray(y_bbvert_pred_sq_raw[b], dtype=np.float16)
        #bbscore_pred_raw = np.asarray(y_bbscore_pred_sq_raw[b], dtype=np.float16)
        #pmask_pred_raw = np.asarray(y_pmask_pred_sq_raw[b], dtype=np.float16)

        Plot.draw_pc(np.concatenate([pc[9:12], pc[3:6]], axis=1))
        Plot.draw_pc_semins(pc_xyz=pc[9:12], pc_semins=ins_gt)
        Plot.draw_pc_semins(pc_xyz=pc[9:12], pc_semins=ins_pred)
        Plot.draw_pc_semins(pc_xyz=pc[9:12], pc_semins=sem_gt)
        Plot.draw_pc_semins(pc_xyz=pc[9:12], pc_semins=sem_pred)

        sub.bat_pc = None
        print("done")

