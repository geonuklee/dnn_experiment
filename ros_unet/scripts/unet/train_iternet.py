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


def compute_loss(target, y1, y2, y3, spliter, validmask, outline_dist):
    fn_w, fp_w = get_w_from_pixel_distribution(target, lamb=5.)
    #fn_w, fp_w = 20., .01
    target = target.float()
    loss1 = masked_loss(spliter, y1, target, validmask, outline_dist, fn_w, fp_w)
    loss2 = masked_loss(spliter, y2, target, validmask, outline_dist, fn_w, fp_w)
    loss3 = masked_loss(spliter, y3, target, validmask, outline_dist, fn_w, fp_w)
    lambda1, lambda2, lambda3 = 1e-1, 2e-1, 3e-1
    loss  = lambda1*loss1 + lambda2*loss2 + lambda3*loss3
    return loss

class TrainEvaluator:
    def __init__(self, valid_dataset, model):
        self.dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        self.model = model

    def save_prediction(self, name):
        vis_dir = 'weights/vis_%s'%name
        if osp.exists(vis_dir):
            shutil.rmtree(vis_dir)
        makedirs(vis_dir)
        for i, data in enumerate(self.dataloader):
            input_x = data['input_x']
            y1, y2, y3 = self.model(input_x)
            y3 =  self.model.spliter.restore(y3)
            dst = self.model.spliter.pred2dst(y3, data['rgb'].squeeze(0).numpy())
            fn = osp.join(vis_dir, '%d.png'%i)
            cv2.imwrite( fn, dst )

    def evaluate_with_valid(self, save_prediction):
        loss_sum = 0.
        if save_prediction:
            vis_dir = 'weights/vis%d'%self.model.niter
            if osp.exists(vis_dir):
                shutil.rmtree(vis_dir)
            makedirs(vis_dir)

        for i, data in enumerate(self.dataloader):
            input_x = data['input_x']
            output = self.model(input_x)
            loss = self.model.ComputeLoss(output, data)
            loss_sum += loss.item()
            y3 = self.model.spliter.restore(output[-1])
            if not save_prediction:
                continue
            dst = self.model.spliter.pred2dst(y3, data['rgb'].squeeze(0).numpy())
            fn = osp.join(vis_dir, '%d.png'%i)
            cv2.imwrite( fn, dst )

        mean_loss = loss_sum / float(len(self.dataloader))
        self.model.valid_loss_list.append( (self.model.niter, mean_loss) )

        if len(self.model.valid_loss_list) < 2:
            return mean_loss
        dtype = [('niter',int), ('loss',float)]
        loss_arr = np.array(self.model.valid_loss_list,dtype=dtype)
        if not hasattr(self, 'fig'):
            self.fig = plt.figure(figsize=(4, 3))
            plt.title('niter - mean loss(valid)')
        else:
            plt.clf()
        fig = self.fig
        plt.plot(loss_arr['niter'], loss_arr['loss'], 'b-')
        plt.savefig('weights/loss_chart.png')
        return mean_loss

def train():
    if not osp.exists('weights'):
        makedirs('weights')
    checkpoint_fn = 'weights/iternet.pth'
    state0 = None
    try:
        state0 = torch.load('weights_big/iternet_min.pth')
    except:
        pass
    #model_name = 'IterNetInterface' #TODO
    #model_name = 'WeightedIterNetInterface'
    model_name = 'BigIterNetInterface' # Good after 6000 iter
    #model_name = 'WeighteBigdIterNetInterface' # Poor than non weighted
    model = globals()[model_name]()

    optimizer = model.CreateOptimizer()
    if state0 is not None:
        model0 = globals()[state0['model_name']]()
        model0.load_state_dict(state0['model_state_dict'])
        model.iternet = model0.iternet
        del model0

        #optimizer.load_state_dict(state0['optimizer_state_dict'])
        if  state0['model_name'] == model.__class__.__name__:
            model.epoch, model.niter, model.min_valid_loss, model.valid_loss_list\
                    = state0['others']

        del state0
        print ("Start with previou weight, epoch last = %d" % model.epoch)
    else:
        print ("Start without previou weight")

    validset = ObbDataset('obb_dataset_alignedroll',augment=False,max_frame_per_scene=3)
    evaluator = TrainEvaluator(validset, model)

    n_epoch = 5
    for model.epoch in range(model.epoch+1, n_epoch):  # loop over the dataset multiple times
        dataset = ObbDataset('obb_dataset_train',augment=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        for i, data in enumerate(dataloader):
            model.iternet.train()
            optimizer.zero_grad(set_to_none=True)
            input_x = data['input_x']
            output = model(input_x)
            loss = model.ComputeLoss(output, data)
            del data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("niter %d, frame %d/%d"%(model.niter, i,len(dataloader) ), end='\r')
            if (model.niter%100 == 0 and model.niter > 0) or i == len(dataloader)-1:
                states = {
                    'model_name' : model.__class__.__name__,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'others' : (model.epoch, model.niter,
                        model.min_valid_loss, model.valid_loss_list)
                }
                current_time = datetime.now().strftime("%H:%M:%S")
                with torch.no_grad():
                    model.eval()
                    save_point = model.niter%500==0
                    valid_loss = evaluator.evaluate_with_valid(save_point)
                    print("epoch [%d/%d], frame[%d/%d], loss(train)= %f, loss(valid)=%f" \
                            % (model.epoch,n_epoch,i,len(dataloader),
                                loss.item(), valid_loss), current_time )
                    if save_point:
                        torch.save(states, 'weights/iternet_%d.pth'%model.niter)
                    if (model.min_valid_loss is None) or valid_loss < model.min_valid_loss:
                        model.min_valid_loss = valid_loss
                        torch.save(states, 'weights/iternet_min.pth')
                        evaluator.save_prediction('minloss')
                    torch.save(states, checkpoint_fn)
                del states
            model.niter += 1

if __name__ == '__main__':
    train()
