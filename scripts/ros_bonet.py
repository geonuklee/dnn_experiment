#!/usr/bin/python2
#-*- coding:utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import numpy as np

import tensorflow as tf
from bonet import BoNet, Data_Configs, Plot, Eval_Tools
from os import path as osp

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
        #import pdb; pdb.set_trace()
        #for pt_each in xyz:
        #    print(pt_each)

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
    script_fn = osp.abspath(__file__)
    pkg_dir = str('/').join(script_fn.split('/')[:-2])
    model_path = osp.join(pkg_dir,'scripts','3D-BoNet','model_released','model.cptk')
    if not osp.isfile(model_path+'.data-00000-of-00001'):
        print(model_path)
        print ('please download the released model!')
        exit(0)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.visible_device_list = '0'
    net.sess = tf.Session(config=config)
    tf.train.Saver().restore(net.sess, model_path)
    print('Model restored sucessful!')

    #gap = 5e-3
    #volume_num = int(1. / gap) + 2
    #volume = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
    #volume_sem = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
    b = 0

    while not rospy.is_shutdown():
        rate.sleep()
        if sub.bat_pc is None:
            continue
        bat_pc = sub.bat_pc
        [y_psem_pred_sq_raw, y_bbvert_pred_sq_raw, y_bbscore_pred_sq_raw, y_pmask_pred_sq_raw] \
            = net.sess.run([net.y_psem_pred,
                net.y_bbvert_pred_raw,
                net.y_bbscore_pred_raw,
                net.y_pmask_pred_raw],
                feed_dict={net.X_pc: bat_pc[:, :, 0:9], net.is_train: False})

        pc = np.asarray(bat_pc[b], dtype=np.float16)

        sem_pred_raw = np.asarray(y_psem_pred_sq_raw[b], dtype=np.float16)
        sem_pred = np.argmax(sem_pred_raw, axis=-1)

        unique_sem = np.unique(sem_pred)
        unique_semnames = [ Data_Configs.sem_names[idx] for idx in unique_sem ]

        bbscore_pred_raw = np.asarray(y_bbscore_pred_sq_raw[b], dtype=np.float16)
        pmask_pred_raw = np.asarray(y_pmask_pred_sq_raw[b], dtype=np.float16)
        pmask_pred \
            = pmask_pred_raw * np.tile(bbscore_pred_raw[:, None], [1, pmask_pred_raw.shape[-1]])
        ins_pred = np.argmax(pmask_pred, axis=-2)

        print("RGB input")
        Plot.draw_pc( np.concatenate([pc[:,9:12], pc[:,6:9]], axis=1) )
        print("Semantic segmentation", unique_semnames)
        Plot.draw_pc_semins(pc_xyz=pc[:,9:12], pc_semins=sem_pred , fix_color_num=13)
        print("Instance segmentation")
        Plot.draw_pc_semins(pc_xyz=pc[:,9:12], pc_semins=ins_pred)

        sub.bat_pc = None
        print("done")

