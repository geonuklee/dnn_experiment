#!/usr/bin/python3
#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pickle
from sys import version_info


if __name__ == '__main__':
    if version_info.major == 3:
        arg={'encoding':'latin1'}
    else:
        arg={}

    picks_bonet = {}
    fn = 'bonet_train_log_vtk_dataset_nooffset.pick'
    with open(fn,'rb') as f:
        picks_bonet['3D-BoNet/close instances'] = pickle.load(f, **arg)

    fn = 'bonet_train_log_vtk_dataset_separated.pick'
    with open(fn,'rb') as f:
        picks_bonet['3D-BoNet/separated instances'] = pickle.load(f, **arg)

    picks_iternet = {}
    fn = 'unet_train_log_vtk_dataset_nooffset.pick'
    with open(fn,'rb') as f:
        picks_iternet['IterNet/close instances'] = pickle.load(f, **arg)
    fn = 'unet_train_log_vtk_dataset_separated.pick'
    with open(fn,'rb') as f:
        picks_iternet['IterNet/separated instances'] = pickle.load(f, **arg)

    all_picks = {}
    for picks in [picks_bonet, picks_iternet, all_picks]:
        for k, v in picks.items():
            all_picks[k] = v

    fontsize = 15
    for i, picks in enumerate([picks_bonet, picks_iternet, all_picks]):
        fig = plt.figure(i, figsize=(8,6), dpi=100)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        for name, pick in picks.items():
            plt.plot(pick['ep'],pick['train_ap'], '-*', linewidth=1, label=name+'/train set' )
            plt.plot(pick['ep'],pick['valid_ap'], '-.', linewidth=1, label=name+'/valid set' )
        plt.ylim(-.1, 1.1)
        plt.legend(fontsize=fontsize,loc='best')
        fig.axes[0].set_xlabel('Epoch', fontsize=fontsize)
        fig.axes[0].set_ylabel('AP', fontsize=fontsize)
        fig.canvas.draw()
        plt.savefig('ap_fig%d.png'%i)
    plt.show()




