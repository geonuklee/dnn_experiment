#!/usr/bin/python3
#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pickle
from sys import version_info
import numpy as np

import cv2
from mpl_toolkits.mplot3d import Axes3D

def draw_unet_bonet_comparison():
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

def draw_watershed():
    heightmap = cv2.imread('/home/geo/.ros/heightmap.png')[:,:,0]
    xx, yy = np.mgrid[0:heightmap.shape[0], 0:heightmap.shape[1]]
    fig = plt.figure(figsize=(12,6),dpi=100)
    ax1 = fig.add_subplot(121, projection='3d')
    # Make panes transparent
    ax1.xaxis.pane.fill = False # Left pane
    ax1.yaxis.pane.fill = False # Right pane
    # Remove grid lines
    ax1.grid(False)
    # Remove tick labels
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    # Transparent spines
    ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # Transparent panes
    ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # No ticks
    ax1.set_xticks([]) 
    ax1.set_yticks([]) 
    ax1.set_zticks([])
    ax1.axis('equal')
    ax1.set_aspect('equal')
    #cmap = plt.cm.gray
    cmap = plt.cm.gray.reversed()
    ax1.plot_surface(xx, yy, heightmap.max()-heightmap, rstride=4, cstride=4,
            cmap=cmap,
            linewidth=0, antialiased=False)
    ax1.view_init(elev=75., azim=0.)

    plt.show()
    print(ax.elev, ax.azim)

    return


if __name__ == '__main__':
    #draw_unet_bonet_comparison()
    draw_watershed()



