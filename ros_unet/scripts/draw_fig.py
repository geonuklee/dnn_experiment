#!/usr/bin/python3
#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pickle
from sys import version_info
import numpy as np


if __name__ == '__main__':
    if version_info.major == 3:
        arg={'encoding':'latin1'}
    else:
        arg={}

    picks = {}
    for fullmethod,method in {'U-Net':'unet', '3D-BoNet':'bonet'}.items():
        for offset in ['offset', 'nooffset']:
            for fulltexture, texture in {'texture':'texture', 'no texture':'notexture'}.items():
                fn = '%s_train_log_vtk_%s_%s.pick' % (method, offset, texture)
                with open(fn,'rb') as f:
                    picks['%s/%s/%s'%(fullmethod,offset,fulltexture)] = pickle.load(f, **arg)

    fontsize = 15
    for method in ['U-Net', '3D-BoNet']:
        fig = plt.figure(figsize=(12,6), dpi=100)
        plt.subplot(1,4,1)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        for offset in ['offset', 'nooffset']:
            for fulltexture in ['texture','no texture']:
                name = '%s/%s/%s'%(method,offset,fulltexture)
                pick = picks[name]
                plt.plot(pick['ep'],pick['train_ap'], '-', linewidth=1, label=name+'/train set' )
                plt.plot(pick['ep'],pick['valid_ap'], '-', linewidth=1, label=name+'/valid set' )

                print(name+'/train set : AP(min IoU=%f) = '%pick['min_iou'],np.max(pick['train_ap'][:150]))
                print(name+'/valid set : AP(min IoU=%f) = '%pick['min_iou'],np.max(pick['valid_ap'][:150]))
        fig.axes[0].set_xlabel('Epoch', fontsize=fontsize)
        fig.axes[0].set_ylabel('AP', fontsize=fontsize)
        leg = plt.legend(fontsize=fontsize, bbox_to_anchor=(1., 1.))
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
        plt.subplots_adjust(right=2.)
        plt.savefig('fig_%s.png'%method)

    #plt.show()


    #fontsize = 15
    #for i, picks in enumerate([picks_bonet, picks_iternet, all_picks]):
    #    fig = plt.figure(i, figsize=(8,6), dpi=100)
    #    plt.xticks(fontsize=fontsize)
    #    plt.yticks(fontsize=fontsize)
    #    for name, pick in picks.items():
    #        plt.plot(pick['ep'],pick['train_ap'], '-*', linewidth=1, label=name+'/train set' )
    #        plt.plot(pick['ep'],pick['valid_ap'], '-.', linewidth=1, label=name+'/valid set' )
    #    plt.ylim(-.1, 1.1)
    #    plt.legend(fontsize=fontsize,loc='best')
    #    fig.axes[0].set_xlabel('Epoch', fontsize=fontsize)
    #    fig.axes[0].set_ylabel('AP', fontsize=fontsize)
    #    fig.canvas.draw()
    #    plt.savefig('ap_fig%d.png'%i)
    #plt.show()




