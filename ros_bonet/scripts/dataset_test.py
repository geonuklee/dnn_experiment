#!/usr/bin/python2
#-*- coding:utf-8 -*-

from dataset_train import *

if __name__ == '__main__':
    #from bonet import BoNet, Plot, Eval_Tools, train
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use
    configs = Data_Configs()
    net = BoNet(configs = configs)
    net.creat_folders(name='log', re_train=False)
    net.build_graph()

    pkg_dir = get_pkg_dir()
    dataset_dir ='/home/docker/obb_dataset_train'
    data = Data_ObbDataset(dataset_dir)

    for i in range(data.total_train_batch_num):
        bat_pc, _, _, bat_psem_onehot, bat_bbvert, bat_pmask = data.load_train_next_batch()
        _, ls_psemce, ls_bbvert_all, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask\
                = net.sess.run([
                    net.optim, net.psemce_loss, net.bbvert_loss, net.bbvert_loss_l2, net.bbvert_loss_ce,
                    net.bbvert_loss_iou,net.bbscore_loss, net.pmask_loss],
                    feed_dict={net.X_pc:bat_pc[:, :, 0:9], net.Y_bbvert:bat_bbvert, net.Y_pmask:bat_pmask,
                        net.Y_psem:bat_psem_onehot, net.lr:l_rate, net.is_train:False})
