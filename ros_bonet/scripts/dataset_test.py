#!/usr/bin/python2
#-*- coding:utf-8 -*-

from dataset_train import *
import tensorflow as tf
from bonet import BoNet, Plot, Eval_Tools, train
import matplotlib.pyplot as plt
import shutil
from os import makedirs
import argparse

def restore_net(model_path, configs):
    #from bonet import BoNet, Plot, Eval_Tools, train
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use
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

    fn = model_path + '.data-00000-of-00001'
    if not os.path.isfile(fn):
        print ('please train the model! Can\'t find %s'%fn )
        exit(1)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.visible_device_list = '0'
    net.sess = tf.Session(config=config)
    tf.train.Saver().restore(net.sess, model_path)
    print('Model restored sucessful!')
    return net

def bbvert2edges(bbvert):
    box = []
    for kz in range(2):
        edges = []
        edges.append( (bbvert[0,0], bbvert[0,1], bbvert[kz,2]) )
        edges.append( (bbvert[1,0], bbvert[0,1], bbvert[kz,2]) )
        edges.append( (bbvert[1,0], bbvert[1,1], bbvert[kz,2]) )
        edges.append( (bbvert[0,0], bbvert[1,1], bbvert[kz,2]) )
        edges.append( (bbvert[0,0], bbvert[0,1], bbvert[kz,2]) )
        ar = np.array(edges)
        box.append(ar)

    edges = []
    edges.append( (bbvert[0,0], bbvert[0,1], bbvert[0,2]) )
    edges.append( (bbvert[0,0], bbvert[0,1], bbvert[1,2]) )
    box.append(np.array(edges))

    edges = []
    edges.append( (bbvert[1,0], bbvert[0,1], bbvert[0,2]) )
    edges.append( (bbvert[1,0], bbvert[0,1], bbvert[1,2]) )
    box.append(np.array(edges))

    edges = []
    edges.append( (bbvert[1,0], bbvert[1,1], bbvert[0,2]) )
    edges.append( (bbvert[1,0], bbvert[1,1], bbvert[1,2]) )
    box.append(np.array(edges))

    edges = []
    edges.append( (bbvert[0,0], bbvert[1,1], bbvert[0,2]) )
    edges.append( (bbvert[0,0], bbvert[1,1], bbvert[1,2]) )
    box.append(np.array(edges))
    return box

def get_colors(pc_semins):
    ins_colors = Plot.random_colors(len(np.unique(pc_semins))+1, seed=2)
    semins_labels = np.unique(pc_semins)
    semins_bbox = []
    Y_colors = np.zeros((pc_semins.shape[0], 3))
    for id, semins in enumerate(semins_labels):
        valid_ind = np.argwhere(pc_semins == semins)[:, 0]
        if semins<=-1:
            tp=[0,0,0]
        else:
            tp = ins_colors[id]
        Y_colors[valid_ind] = tp
    return Y_colors

if __name__ == '__main__':
    from vtk_train import Data_VtkDataset
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, nargs='?', help='The name of output dataset')
    args = parser.parse_args()
    if args.dataset_name==None:
        args.dataset_name = 'vtk_dataset'

    model_path = 'log-%s/train_mod/model.cptk'%args.dataset_name
    configs = Data_Configs()
    net = restore_net(model_path, configs)

    data = Data_VtkDataset(args.dataset_name+'/test', batch_size=1)
    pred, recall = evaluation(net, data, configs, min_iou=.5)
    print("pred = ", pred)

    data = Data_VtkDataset(args.dataset_name+'/test', batch_size=1)
    bonet_output_dir = osp.join(args.dataset_name, 'bonet_output')
    if osp.exists(bonet_output_dir):
        shutil.rmtree(bonet_output_dir)
    makedirs(bonet_output_dir)

    for i in range(data.scene_num):
        bat_pc, bat_sem_gt, bat_ins_gt, bat_psem_onehot, bat_bbvert, bat_pmask, rgb \
                = data.load_next_scene()
        # pc_all.shape = N,12 , ins_gt_all.shape = N,
        pc_all = []; ins_gt_all = []; sem_pred_all = []; sem_gt_all = []
        gap = .01 #gap = 5e-3
        volume_num = int(1. / gap) + 2
        volume = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
        volume_sem = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
        X_pc = bat_pc

        [y_psem_pred_sq_raw, y_bbvert_pred_raw, y_bbscore_pred_sq_raw, y_pmask_pred_sq_raw] = \
            net.sess.run([net.y_psem_pred, net.y_bbvert_pred_raw, net.y_bbscore_pred_raw, net.y_pmask_pred_raw],feed_dict={net.X_pc: X_pc[:, :, 0:9], net.is_train: False})

        for b in range(bat_pc.shape[0]):
            pc = bat_pc[b].astype(np.float16)
            sem_gt = bat_sem_gt[b].astype(np.int16)
            ins_gt = bat_ins_gt[b].astype(np.int32)

            sem_pred_raw = np.asarray(y_psem_pred_sq_raw[b], dtype=np.float16)
            bbvert_pred_raw = np.asarray(y_bbvert_pred_raw[b], dtype=np.float16)
            bbscore_pred_raw = y_bbscore_pred_sq_raw[b].astype(np.float16)
            pmask_pred_raw = y_pmask_pred_sq_raw[b].astype(np.float16)

            sem_pred = np.argmax(sem_pred_raw, axis=-1)
            pmask_pred = pmask_pred_raw * np.tile(bbscore_pred_raw[:, None], [1, pmask_pred_raw.shape[-1]])
            ins_pred = np.argmax(pmask_pred, axis=-2)
            ins_sem_dic = Eval_Tools.get_sem_for_ins(ins_by_pts=ins_pred, sem_by_pts=sem_pred)
            Eval_Tools.BlockMerging(volume, volume_sem, pc[:, :3], ins_pred, ins_sem_dic, gap)

            bbvert_err = (bbvert_pred_raw - bat_bbvert[b])[np.unique(ins_pred)]
            max_bbvert_err = np.abs(bbvert_err).max()

            pc_all.append(pc)
            ins_gt_all.append(ins_gt)
            sem_pred_all.append(sem_pred)
            sem_gt_all.append(sem_gt)

        pc_all = np.concatenate(pc_all, axis=0)
        ins_gt_all = np.concatenate(ins_gt_all, axis=0)
        sem_pred_all = np.concatenate(sem_pred_all, axis=0)
        sem_gt_all = np.concatenate(sem_gt_all, axis=0)

        pc_xyz_int = (pc_all[:, :3] / gap).astype(np.int32)
        ins_pred_all = volume[tuple(pc_xyz_int.T)]

        #print('ins_gt=', np.unique(ins_gt_all)) # TODO << 이게 왜 0,1,2가 아니라..
        #print('ins_pred=', np.unique(ins_pred_all))
        #print('sem_gt=', np.unique(sem_gt_all))
        #print('sem_pred=', np.unique(sem_pred_all))
        #continue

        sub_rows, sub_cols = 3, bat_pc.shape[0]
        fig = plt.figure(1, figsize=(8,7), dpi=100)
        fig.clf()
        ax = fig.add_subplot(2,2,1, projection='3d')
        ax.scatter(pc_all[:,-3],pc_all[:, -2],pc_all[:, -1], c=pc_all[:,3:6], s=4, linewidths=0)
        #ax.axis('off')
        ax.view_init(elev=-90., azim=-90)
        ax.set_zticks([])
        plt.axis('equal')

        colors = get_colors(ins_gt_all)
        ax = fig.add_subplot(2,2,2, projection='3d')
        ax.scatter(pc_all[:,-3],pc_all[:,-2],pc_all[:,-1], c=colors, s=2, linewidths=0)
        ax.view_init(elev=-90., azim=-90)
        ax.axis('off')
        plt.axis('equal')

        vec = np.stack((ins_gt_all,ins_pred_all)).T
        matches, indices, counts = np.unique(vec,axis=0,return_counts=True,return_inverse=True)
        for i, n in enumerate(counts):
            if n < Data_Configs.train_pts_num * 0.1 :
                continue
            #pair = matches[i]
            pred = pc_all[indices==i][:,:3]
            gt   = pc_all[ins_gt_all == matches[i][0] ][:,:3]
            pred_bb = np.stack( (np.amin(pred,axis=0),np.amax(pred,axis=0) ) )
            gt_bb = np.stack( (np.amin(gt,axis=0),np.amax(gt,axis=0) ) )
            bb_err = np.abs( pred_bb - gt_bb ).max()

        colors = get_colors(ins_pred_all)
        ax = fig.add_subplot(2,2,3, projection='3d')
        ax.scatter(pc_all[:,0],pc_all[:,1],pc_all[:,2], c=colors, s=2, linewidths=0)
        for i, ins in enumerate( np.unique(ins_pred) ):
            edges = bbvert2edges( bbvert_pred_raw[ins] )
            color = colors[i]
            for edge in edges:
                ax.plot(edge[:,0], edge[:,1], edge[:,2], c='b')

        for i, ins in enumerate( np.unique(ins_pred) ):
            edges = bbvert2edges( bbvert_pred_raw[ins] )
            color = colors[i]
            for edge in edges:
                ax.plot(edge[:,0], edge[:,1], edge[:,2], c='b')

        ax.view_init(elev=-90., azim=-90)
        ax.axis('off')
        plt.axis('equal')

        ax = fig.add_subplot(2,2,4)
        ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), extent = [0,1,0,1],
                aspect=float(rgb.shape[0])/float(rgb.shape[1]) )
        plt.axis('off')
        plt.axis('equal')


        #fig = plt.figure(2, figsize=(15,10), dpi=80)
        #fig.clf()
        #for b in range(bat_pc.shape[0]):
        #    if b > sub_cols:
        #        break
        #    pc = bat_pc[b].astype(np.float16)
        #    sem_gt = bat_sem_gt[b].astype(np.int16)
        #    ins_gt = bat_ins_gt[b].astype(np.int32)
        #    sem_pred_raw = np.asarray(y_psem_pred_sq_raw[0], dtype=np.float16)
        #    bbvert_pred_raw = np.asarray(y_bbvert_pred_raw[0], dtype=np.float16)
        #    bbscore_pred_raw = y_bbscore_pred_sq_raw[0].astype(np.float16)
        #    pmask_pred_raw = y_pmask_pred_sq_raw[0].astype(np.float16)
        #    sem_pred = np.argmax(sem_pred_raw, axis=-1)
        #    pmask_pred = pmask_pred_raw * np.tile(bbscore_pred_raw[:, None], [1, pmask_pred_raw.shape[-1]])
        #    ins_pred = np.argmax(pmask_pred, axis=-2)

        #    ax = fig.add_subplot(sub_rows,sub_cols,b+1, projection='3d')
        #    ax.scatter(pc[:,-3],pc[:, -2],pc[:, -1], c=pc[:,3:6], s=2, linewidths=0)
        #    ax.view_init(elev=-90., azim=-90)
        #    ax.axis('off')

        #    colors = get_colors(ins_gt)
        #    ax = fig.add_subplot(sub_rows,sub_cols,sub_cols+b+1, projection='3d')
        #    ax.scatter(pc[:,-3],pc[:, -2],pc[:, -1], c=colors, s=2, linewidths=0)
        #    ax.view_init(elev=-90., azim=-90)
        #    ax.axis('off')

        #    colors = get_colors(ins_pred)
        #    ax = fig.add_subplot(sub_rows,sub_cols,2*sub_cols+b+1, projection='3d')
        #    ax.scatter(pc[:,-3],pc[:, -2],pc[:, -1], c=colors, s=2, linewidths=0)
        #    ax.view_init(elev=-90., azim=-90)
        #    ax.axis('off')
        print("Draw")
        fig.canvas.draw()
        plt.show(block=True)
