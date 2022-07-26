#!/usr/bin/python2
#-*- coding:utf-8 -*-

"""!
@file vtk_train.py
@brief Train 3D-Bonet with vtk_dataset to proove perfomance at alinged boxes.

The vtk_dataset generate from gen_vtkscene.py->gen_ponitcloud.py of ros_unet

"""

import numpy as np
import tensorflow as tf
from bonet import BoNet, Plot, Eval_Tools
from os import path as osp
import os
import copy
from os import listdir
from indoor3d_util import room2blocks
import random
import glob2
import shutil
from bonet_dataset.dataset import Data_VtkDataset, Data_ObbDataset, Data_Configs, get_blocks, xyz_normalize
from dataset_train import train
from os import makedirs
import pickle
from os import path as osp
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, nargs='?', help='The name of output dataset')
    parser.add_argument('--dev', metavar='N', type=int, nargs='?', help='The device')
    args = parser.parse_args()
    if args.dataset_name==None:
        args.dataset_name = 'vtk_dataset'
    if args.dev==None:
        args.dev = 0
    script_fn = osp.abspath(__file__)
    pkg_dir = str('/').join(script_fn.split('/')[:-2])
    train_dataset = Data_VtkDataset(args.dataset_name+'/train', batch_size=4)
    valid_dataset = Data_VtkDataset(args.dataset_name+'/test', batch_size=1)
    train_dataset2 = Data_VtkDataset(args.dataset_name+'/train', batch_size=1)
    log_name = 'log-'+args.dataset_name

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '%d'%args.dev  ## specify the GPU to use
    configs = Data_Configs()
    net = BoNet(configs = configs)
    net.creat_folders(name=log_name, re_train=False)
    shutil.copyfile(__file__, osp.join(log_name,'vtk_train.py') )
    shutil.copyfile(osp.join(pkg_dir, 'scripts','dataset_train.py'), osp.join(log_name,'dataset_train.py') )
    shutil.copyfile(args.dataset_name+'/gen_vtkscene.py', osp.join(log_name,'gen_vtkscene.py') )
    net.build_graph()
    train(net, train_dataset, valid_dataset, train_dataset2, configs)


