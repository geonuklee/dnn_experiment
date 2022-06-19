#!/usr/bin/python2
#-*- coding:utf-8 -*-

from dataset_train import *
import tensorflow as tf
from bonet import BoNet, Plot, Eval_Tools, train

def restore_net():
    #from bonet import BoNet, Plot, Eval_Tools, train
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use
    configs = Data_Configs()
    net = BoNet(configs = configs)
    ####### 1. networks
    net.X_pc = tf.placeholder(shape=[None, None, net.points_cc], dtype=tf.float32, name='X_pc')
    net.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
    with tf.variable_scope('backbone'):
        #net.point_features, net.global_features, net.y_psem_pred = net.backbone_pointnet(net.X_pc, net.is_train)
        net.point_features, net.global_features, net.y_psem_pred = net.backbone_pointnet2(net.X_pc, net.is_train)
    with tf.variable_scope('bbox'):
        net.y_bbvert_pred_raw, net.y_bbscore_pred_raw = net.bbox_net(net.global_features)
    with tf.variable_scope('pmask'):
        net.y_pmask_pred_raw = net.pmask_net(net.point_features, net.global_features, net.y_bbvert_pred_raw, net.y_bbscore_pred_raw)

    model_path = 'log/train_mod/model050.cptk'
    if not os.path.isfile(model_path + '.data-00000-of-00001'):
        print ('please download the released model!')
        return
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.visible_device_list = '0'
    net.sess = tf.Session(config=config)
    tf.train.Saver().restore(net.sess, model_path)
    print('Model restored sucessful!')
    return net


if __name__ == '__main__':
    net = restore_net()
    pkg_dir = get_pkg_dir()
    dataset_dir ='/home/docker/obb_dataset_train'
    data = Data_ObbDataset(dataset_dir, batch_size=1)

    for i in range(data.total_train_batch_num):
        bat_pc, bat_sem_gt, bat_ins_gt, bat_psem_onehot, bat_bbvert, bat_pmask = data.load_train_next_batch()
        [y_psem_pred_sq_raw, y_bbvert_pred_sq_raw, y_bbscore_pred_sq_raw, y_pmask_pred_sq_raw] = \
                net.sess.run([net.y_psem_pred, net.y_bbvert_pred_raw, net.y_bbscore_pred_raw, net.y_pmask_pred_raw],feed_dict={net.X_pc: bat_pc[:, :, 0:9], net.is_train: False})
        # TODO concatenate for each file
        # 1) Visualization
        # pc_all.shape = N,12 , ins_gt_all.shape = N,
        for b in range(bat_pc.shape[0]):
            pc_all = bat_pc[b].astype(np.float16)
            sem_gt = bat_sem_gt[b].astype(np.int16)
            ins_gt = bat_ins_gt[b].astype(np.int32)
            bbscore_pred_raw = y_bbscore_pred_sq_raw[b].astype(np.float16)
            pmask_pred_raw = y_pmask_pred_sq_raw[b].astype(np.float16)

            pmask_pred = pmask_pred_raw * np.tile(bbscore_pred_raw[:, None], [1, pmask_pred_raw.shape[-1]])
            ins_pred = np.argmax(pmask_pred, axis=-2)
            #ins_sem_dic = Eval_Tools.get_sem_for_ins(ins_by_pts=ins_pred, sem_by_pts=sem_pred)
            Plot.draw_pc(np.concatenate([pc_all[:,9:12], pc_all[:,3:6]], axis=1))
            Plot.draw_pc_semins(pc_xyz=pc_all[:, 9:12], pc_semins=ins_pred)
            # 2) Merging block
            # 3) Static score - if required
