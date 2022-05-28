#!/usr/bin/python3
#-*- coding:utf-8 -*-

from os import path as osp
import sys
import pickle
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch.nn.functional as F
import glob2
import cv2
import numpy as np
from segment_dataset import ObbDataset
from util import *
#from unet_model import IterNet
from iternet import *
from torch import nn, optim
from datetime import datetime
import matplotlib.pyplot as plt
from os import makedirs
import shutil


def spliter_test():
    dataset = ObbDataset('obb_dataset_train')
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for data in loader:
        break
    rgb = data['rgb']
    # b,c,h,w
    rgb = rgb.moveaxis(-1,1)

    spliter = SplitAdapter(wh = 200,step = 120)
    patches = spliter.put(rgb)
    dst = spliter.restore(patches)

    cv2.imshow("dst", dst[0,:,:,:].moveaxis(0,-1).numpy())
    cv2.waitKey()

def compute_loss(target, y1, y2, y3, spliter, validmask, outline_dist, f_loss=masked_loss, lamb=5.):
    fn_w, fp_w = get_w_from_pixel_distribution(target, lamb)
    #fn_w, fp_w = 20., .01
    target = target.float()
    loss1 = masked_loss(spliter, y1, target, validmask, outline_dist, fn_w, fp_w)
    loss2 = masked_loss(spliter, y2, target, validmask, outline_dist, fn_w, fp_w)
    loss3 = masked_loss(spliter, y3, target, validmask, outline_dist, fn_w, fp_w)
    lambda1, lambda2, lambda3 = 1e-1, 2e-1, 3e-1
    loss  = lambda1*loss1 + lambda2*loss2 + lambda3*loss3
    return loss

class TrainEvaluator:
    def __init__(self, valid_dataset, valid_loss_list=[]):
        self.dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        self.valid_loss_list = valid_loss_list

    def put_train_loss(self, loss):
        pass

    def save_prediction(self, spliter, model, device, name):
        vis_dir = 'weights/vis_%s'%name
        if osp.exists(vis_dir):
            shutil.rmtree(vis_dir)
        makedirs(vis_dir)
        for i, data in enumerate(self.dataloader):
            input_x = data['input_x']
            input_x = spliter.put(input_x).to(device)
            _, _, y3 = model(input_x)
            y3 = spliter.restore(y3)
            dst = spliter.pred2dst(y3, data['rgb'].squeeze(0).numpy())
            fn = osp.join(vis_dir, '%d.png'%i)
            cv2.imwrite( fn, dst )

    def evaluate_with_valid(self, spliter, model, device, niter, save_prediction):
        loss_sum = 0.
        if save_prediction:
            vis_dir = 'weights/vis%d'%niter
            if osp.exists(vis_dir):
                shutil.rmtree(vis_dir)
            makedirs(vis_dir)

        for i, data in enumerate(self.dataloader):
            input_x = data['input_x']
            input_x = spliter.put(input_x).to(device)
            y1, y2, y3 = model(input_x)
            target = data['outline']
            target = spliter.put(target)
            outline_dist = data['outline_dist']
            outline_dist = spliter.put(outline_dist)
            validmask = spliter.put(data['validmask'])

            loss = compute_loss(target, y1, y2, y3, spliter, validmask, outline_dist,
                    f_loss=masked_loss2, lamb=100.)

            loss_sum += loss.item()
            y3 = spliter.restore(y3)

            if not save_prediction:
                continue

            dst = spliter.pred2dst(y3, data['rgb'].squeeze(0).numpy())
            fn = osp.join(vis_dir, '%d.png'%i)
            cv2.imwrite( fn, dst )

        mean_loss = loss_sum / float(len(self.dataloader))
        self.valid_loss_list.append( (niter, mean_loss) )

        if len(self.valid_loss_list) < 2:
            return mean_loss
        dtype = [('niter',int), ('loss',float)]
        loss_arr = np.array(self.valid_loss_list,dtype=dtype)
        if not hasattr(self, 'fig'):
            self.fig = plt.figure(figsize=(4, 3))
            plt.title('niter - mean loss(valid)')
        else:
            plt.clf()
        fig = self.fig
        plt.plot(loss_arr['niter'], loss_arr['loss'], 'b-')
        plt.savefig('weights/loss_chart.png')
        #plt.show(block=False)
        #plt.pause(.001)
        return mean_loss

def train():
    spliter = SplitAdapter(256, 200)
    device = "cuda:0"
    model = IterNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    validset = ObbDataset('obb_dataset_alignedroll',augment=False,max_frame_per_scene=3)
    evaluator = TrainEvaluator(validset)

    if not osp.exists('weights'):
        makedirs('weights')

    checkpoint_fn = 'weights/iternet.pth'
    try:
        checkpoint = torch.load('weights_useable_validmask/iternet_min.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_last = checkpoint['epoch']
        niter = checkpoint['niter']
        min_valid_loss = None
        print ("Start with previou weight, epoch last = %d" % epoch_last)
        del checkpoint
    except:
        exit(1)

    n_epoch = epoch_last+6
    for epoch in range(epoch_last+1, n_epoch):  # loop over the dataset multiple times
        dataset = ObbDataset('obb_dataset_train',augment=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        for i, data in enumerate(dataloader):
            input_x = data['input_x']
            input_x = spliter.put(input_x).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            y1, y2, y3 = model(input_x)
            target = spliter.put(data['outline'])
            outline_dist = spliter.put( data['outline_dist'])
            validmask = spliter.put(data['validmask'])

            loss = compute_loss(target, y1, y2, y3, spliter, validmask, outline_dist,
                    f_loss=masked_loss2, lamb=100.)

            #del y1,y2,y3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            states = {
                'epoch': epoch,
                'niter': niter,
                'valid_loss_list':evaluator.valid_loss_list,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }

            print("niter %d, frame %d"%(niter, i), end='\r')
            if (niter%100 == 0 and niter > 0) or i == len(dataloader)-1:
                current_time = datetime.now().strftime("%H:%M:%S")
                print("epoch [%d/%d], frame[%d/%d] loss = %f" \
                        % (epoch,n_epoch,i,len(dataloader), loss.item()),
                        current_time )
                with torch.no_grad():
                    model.eval()
                    save_point = niter%500==0
                    valid_loss = evaluator.evaluate_with_valid(spliter, model, device,
                            niter, save_prediction=save_point)
                    if save_point:
                        torch.save(states, 'weights/iternet_%d.pth'%niter)
                    if (min_valid_loss is None) or valid_loss < min_valid_loss:
                        min_valid_loss = valid_loss
                        states['min_valid_loss'] = valid_loss
                        torch.save(states, 'weights/iternet_min.pth')
                        evaluator.save_prediction(spliter, model, device, 'minloss')
                    torch.save(states, checkpoint_fn)
            niter += 1

if __name__ == '__main__':
    train()
    #test_validation()
