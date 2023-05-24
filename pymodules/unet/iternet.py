'''

https://github.com/hyungminr/PyTorch_IterNet

'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class conv2L(nn.Module):
    def __init__(self, dim_in, dim_out, kernel=3, stride=1, padding=1, bias=True):
        super(conv2L, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
        layers += [nn.BatchNorm2d(num_features=dim_out)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=dim_out, out_channels=dim_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
        layers += [nn.BatchNorm2d(num_features=dim_out)]
        layers += [nn.ReLU(inplace=True)]
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)
    
class conv1L(nn.Module):
    def __init__(self, dim_in, dim_out, kernel=1, stride=1, padding=0, bias=True, activation='sigmoid'):
        super(conv1L, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
        layers += [nn.BatchNorm2d(num_features=dim_out)]
        if activation == 'sigmoid':
            layers += [nn.Sigmoid()]
        else:
            layers += [nn.ReLU(inplace=True)]
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)
    
class maxPool(nn.Module):
    def __init__(self, kernel=2, stride=2, padding=0):
        super(maxPool, self).__init__()
        layers = []
        layers += [nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)]
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)
    
class convUp(nn.Module):
    def __init__(self, dim_in, dim_out, kernel=3, stride=1, padding=1, bias=True):
        super(convUp, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=dim_in, out_channels=dim_out*2, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
        layers += [nn.BatchNorm2d(num_features=dim_out*2)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=dim_out*2, out_channels=dim_out*2, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
        layers += [nn.BatchNorm2d(num_features=dim_out*2)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.ConvTranspose2d(in_channels=dim_out*2, out_channels=dim_out, kernel_size=2, stride=2)]
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)
    
class IterNetModule(nn.Module):
    def __init__(self, n):
        super(IterNetModule, self).__init__()
        self.pool = maxPool()
        self.cd = conv1L(dim_in= 32*n, dim_out= 32, kernel=1, stride=1, padding=0, bias=True, activation='relu')
        self.c0 = conv2L(dim_in= 32, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.c1 = conv2L(dim_in= 32, dim_out= 64, kernel=3, stride=1, padding=1, bias=True)
        self.c2 = conv2L(dim_in= 64, dim_out=128, kernel=3, stride=1, padding=1, bias=True)
        self.c3 = convUp(dim_in=128, dim_out=128, kernel=3, stride=1, padding=1, bias=True)
        self.c4 = convUp(dim_in=256, dim_out= 64, kernel=3, stride=1, padding=1, bias=True)
        self.c5 = convUp(dim_in=128, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.c6 = conv2L(dim_in= 64, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.co = conv1L(dim_in= 32, dim_out=  2, kernel=1, stride=1, padding=0, bias=True)
        
    def forward(self, x0, x6, x9s=None):
        x9 = self.c0(x6)
        if x9s is not None:
            x9_ = torch.cat((x9s, x9), dim=1)
            x9_ = torch.cat((x9_, x0[:,:,:x9_.shape[2],:x9_.shape[3]]), dim=1)
            x9s = torch.cat((x9, x9s), dim=1)
        else:
            x9s = x9
            x9_ = torch.cat((x9, x0[:,:,:x9.shape[2],:x9.shape[3]]), dim=1)
        x0c = self.cd(x9_)
        x1p = self.pool(x0c)
        x1c = self.c1(x1p)
        x2p = self.pool(x1c)
        x2c = self.c2(x2p)
        x3p = self.pool(x2c)

        x3 = self.c3(x3p)
        x3 = F.pad(x3, pad=(0,x2c.shape[-1]-x3.shape[-1],0,x2c.shape[-2]-x3.shape[-2]),
                mode='constant', value=0)
        x3 = torch.cat((x3, x2c), dim=1)

        x4 = self.c4(x3)
        x4 = F.pad(x4, pad=(0,x1c.shape[-1]-x4.shape[-1],0,x1c.shape[-2]-x4.shape[-2]),
                mode='constant', value=0)
        x4 = torch.cat((x4, x1c), dim=1)

        x5 = self.c5(x4)
        x5 = F.pad(x5, pad=(0,x0c.shape[-1]-x5.shape[-1],0,x0c.shape[-2]-x5.shape[-2]),
                mode='constant', value=0)
        x5 = torch.cat((x5, x0c), dim=1)

        x6 = self.c6(x5)
        xout = self.co(x6)
        return xout, x6, x9s
        
class IterNetInit(nn.Module):
    def __init__(self, input_ch):
        '''
        input_ch 3 for hessian, Gx,Gy
        input_ch 4 for gray, hessian, Gx,Gy
        '''
        super(IterNetInit, self).__init__()
        self.block1_pool = maxPool()
        self.block1_c1 = conv2L(dim_in=input_ch, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c2 = conv2L(dim_in= 32, dim_out= 64, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c3 = conv2L(dim_in= 64, dim_out=128, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c4 = conv2L(dim_in=128, dim_out=256, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c5 = convUp(dim_in=256, dim_out=256, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c6 = convUp(dim_in=512, dim_out=128, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c7 = convUp(dim_in=256, dim_out= 64, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c8 = convUp(dim_in=128, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c9 = conv2L(dim_in= 64, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.block1_co = conv1L(dim_in= 32, dim_out=  2, kernel=1, stride=1, padding=0, bias=True)
        
    def forward(self, img):
        x1c = self.block1_c1(img)
        x1p = self.block1_pool(x1c)
        x2c = self.block1_c2(x1p)
        x2p = self.block1_pool(x2c)
        x3c = self.block1_c3(x2p)
        x3p = self.block1_pool(x3c)
        x4c = self.block1_c4(x3p)
        x4p = self.block1_pool(x4c)

        #for i in range(5,9):
        #    j = 9-i
        #    if j == 4:
        #        xp = "x%dp"%(i-1)
        #    else:
        #        xp = "x%d"%(i-1)
        #    xc = "x%dc"%j
        #    cmd = "x%d = self.block1_c%d(%s)"%(i,i,xp)
        #    print(cmd)
        #    cmd = "x%d = F.pad(x%d, pad=(0,%s.shape[-1]-x%d.shape[-1],0,%s.shape[-2]-x%d.shape[-2]),\nmode='constant', value=0)"\
        #        %(i,i,xc,i,xc,i)
        #    print(cmd)
        #    cmd = "x%d = torch.cat( (x%d,%s), dim=1)"%(i,i,xc)
        #    print(cmd)
        #    print('')
        x5 = self.block1_c5(x4p)
        x5 = F.pad(x5, pad=(0,x4c.shape[-1]-x5.shape[-1],0,x4c.shape[-2]-x5.shape[-2]),
                mode='constant', value=0)
        x5 = torch.cat( (x5,x4c), dim=1)
        
        x6 = self.block1_c6(x5)
        x6 = F.pad(x6, pad=(0,x3c.shape[-1]-x6.shape[-1],0,x3c.shape[-2]-x6.shape[-2]),
                mode='constant', value=0)
        x6 = torch.cat( (x6,x3c), dim=1)
        
        x7 = self.block1_c7(x6)
        x7 = F.pad(x7, pad=(0,x2c.shape[-1]-x7.shape[-1],0,x2c.shape[-2]-x7.shape[-2]),
                mode='constant', value=0)
        x7 = torch.cat( (x7,x2c), dim=1)
        
        x8 = self.block1_c8(x7)
        x8 = F.pad(x8, pad=(0,x1c.shape[-1]-x8.shape[-1],0,x1c.shape[-2]-x8.shape[-2]),
                mode='constant', value=0)
        x8 = torch.cat( (x8,x1c), dim=1)

        x9 = self.block1_c9(x8)
        xout = self.block1_co(x9)
        
        return xout, x1c, x9
    

class IterNet(nn.Module):
    def __init__(self,input_ch):
        super(IterNet, self).__init__()
        self.net1 = IterNetInit(input_ch)
        self.net2 = IterNetModule(2)
        self.net3 = IterNetModule(3)
        #self.net4 = IterNetModule(4) # Not enough cuda memory.

    def forward(self, img):
        out1, x0, x9 = self.net1(img)
        out2, x6, x9 = self.net2(x0, x9)
        out3, x6, x9 = self.net3(x0, x6, x9)
        #out4, x6, x9 = self.net4(x0, x6, x9)
        return out1, out2, out3

def get_w_from_pixel_distribution(gt, lamb=1):
    N = (gt == 0).sum()
    P = (gt == 1).sum()
    w1 = 2.*N / (lamb*P + N)
    w2 = 2.*P / (P + lamb*N)
    return w1, w2

def weighted_bce_loss(output, target, w1, w2):
    y = output * target
    loss1 = (w1 - w2) * F.binary_cross_entropy(y, target)
    loss2 = w2 * F.binary_cross_entropy(output, target)
    """
    1) target = 0
    loss1 = (w1-w2) * (0 + log(1)) = 0
    loss2 = w2 * (0 + log(1-a))
    => loss = w2 * (log(1-a))
    
    2) target = 1
    loss1 = (w1-w2) * (log(a) + 0)
    loss2 = w2 * (log(a) + 0)
    => loss = w1 * log(a)
    
    => loss = w1*y*log(a) + w2*(1-y)*log(1-a)
    """
    return loss1 + loss2

class IterNet2(nn.Module):
    def __init__(self,input_ch):
        super(IterNet, self).__init__()
        self.net1 = IterNetInit(input_ch=input_ch)
        self.net2 = IterNetModule(2)
        self.net3 = IterNetModule(3)
        #self.net4 = IterNetModule(4) # Not enough cuda memory.

    def forward(self, img):
        out1, x0, x9 = self.net1(img)
        for i in range(4):
            out, x0, x9 = self.net2(x0, x9)
        return out1, out2, out3

if __package__ is '':
    from util import *
else:
    import sys
    from os import path
    pack_path = path.dirname( path.abspath(__file__) )
    sys.path.append(pack_path)
    from util import *

from torch import nn, optim

class IterNetInterface(nn.Module):
    def __init__(self, input_ch):
        super(IterNetInterface, self).__init__()
        self.device = 'cuda:0'
        self.iternet = IterNet(input_ch).to(self.device)
        self.spliter = SplitAdapter(128, 100)
        self.epoch = -1
        self.niter= 0
        self.min_valid_loss = None
        self.valid_loss_list = []

    def forward(self, input_x):
        input_x = self.spliter.put(input_x).to(self.device)
        y1, y2, y3 = self.iternet(input_x)
        return y1,y2,y3

    def ComputeLoss(self, output, data):
        outline, convex_edges, validmask, outline_dist \
                = data['outline'], data['convex_edges'], data['validmask'], data['outline_dist']
        validmask = torch.cat((validmask,validmask),dim=1)
        #target = outline
        target = torch.cat((outline,convex_edges),dim=1)
        y1,y2,y3=output
        target, validmask, outline_dist \
                = [ self.spliter.put(x) for x in (target, validmask, outline_dist) ]

        #fn_w, fp_w = get_w_from_pixel_distribution(target, lamb=100.)
        fn_w, fp_w = 300., 1.
        #fn_w, fp_w = 150., 1.

        target = target.float()
        loss1 = self.ComputeEachLoss( y1, target, validmask, outline_dist, fn_w, fp_w)
        loss2 = self.ComputeEachLoss( y2, target, validmask, outline_dist, fn_w, fp_w)
        loss3 = self.ComputeEachLoss( y3, target, validmask, outline_dist, fn_w, fp_w)
        lambda1, lambda2, lambda3 = 1e-1, 2e-1, 3e-1
        loss  = lambda1*loss1 + lambda2*loss2 + lambda3*loss3
        return loss

    def ComputeEachLoss(self, output, target, validmask, outline_dist, fn_w, fp_w):
        assert(output.shape[0] == target.shape[0])
        assert(output.shape[1] == 2)
        assert(target.shape[1] == 2)
        if output.device != torch.device('cpu'):
            target = target.to(output.device)
        ignore_border = True
        if ignore_border:
            offset = self.spliter.border
            cropped_output = output[:,:,offset:-offset-1, offset:-offset-1]
            cropped_target = target[:,:,offset:-offset-1, offset:-offset-1]
            cropped_weights = torch.ones_like(cropped_target)
            cropped_weights[validmask[:,:,offset:-offset-1, offset:-offset-1]==0] = 0.
            cropped_weights = cropped_weights.detach()
            yn = cropped_target*cropped_output
            fn_loss  = fn_w * F.binary_cross_entropy( yn,             cropped_target, cropped_weights)
            fnp_loss = fp_w * F.binary_cross_entropy( cropped_output, cropped_target, cropped_weights)
        else:
            yn = target*output 
            fn_loss  = fn_w * F.binary_cross_entropy(yn,     target)
            fnp_loss = fp_w * F.binary_cross_entropy(output, target)
        return fn_loss+fnp_loss

    def CreateOptimizer(self):
        return optim.SGD(self.iternet.parameters(), lr=0.001, momentum=0.9)


class WeightedIterNetInterface(IterNetInterface):
    def __init__(self):
        super(WeightedIterNetInterface, self).__init__()

    def ComputeEachLoss(self, output, target, validmask, outline_dist, fn_w, fp_w):
        assert(output.shape[0] == target.shape[0])
        assert(output.shape[1] == 1)
        assert(target.shape[1] == 1)
        offset = int( (self.spliter.wh-self.spliter.step-2) / 2 )+1
        if output.device != torch.device('cpu'):
            target = target.to(output.device)
        cropped_output = output[:,:,offset:-offset-1, offset:-offset-1]
        cropped_target = target[:,:,offset:-offset-1, offset:-offset-1]

        #### step 1. outline weights
        ksize = self.spliter.wh +1 - cropped_output.shape[-1]
        yn = target*output
        alpha = 1.
        ones = torch.ones( (1,1,ksize,ksize) ).to(output.device)
        t1 = cropped_target * F.conv2d(target-yn, ones)
        t2 = F.conv2d(target, ones)
        t2[cropped_target == 0.] = 1.
        outline_weights = t1 /t2
        outline_weights = outline_weights* (2.-alpha) + alpha # weight : alpha ~ 2.

        #### step 2. non outline weights
        alpha = .2
        cropped_dist = outline_dist[:,:,offset:-offset-1, offset:-offset-1]
        non_outline_weights = cropped_dist/50.
        non_outline_weights[non_outline_weights > 1.] = 1.
        non_outline_weights = non_outline_weights* (1.-alpha) + alpha
        non_outline_weights = non_outline_weights.to(output.device)

        cropped_weights = torch.ones_like(cropped_target)
        b = cropped_dist < 5.
        cropped_weights[b] =     outline_weights[b] # It made output worse
        cropped_weights[~b] = non_outline_weights[~b]
        cropped_weights[validmask[:,:,offset:-offset-1, offset:-offset-1]==0] = 0.
        cropped_weights = cropped_weights.detach()

        yn = yn[:,:,offset:-offset-1, offset:-offset-1]
        fn_loss  = fn_w * F.binary_cross_entropy( yn,             cropped_target, cropped_weights)
        fnp_loss = fp_w * F.binary_cross_entropy( cropped_output, cropped_target, cropped_weights)
        return fn_loss+fnp_loss

class BigIterNetInterface(IterNetInterface):
    def __init__(self, input_ch):
        super(BigIterNetInterface, self).__init__(input_ch)
        self.spliter = SplitAdapter(312, 300)
        #self.spliter = SplitAdapter(126, 100)

class WeighteBigdIterNetInterface(WeightedIterNetInterface):
    def __init__(self):
        super(WeighteBigdIterNetInterface, self).__init__()
        self.spliter = SplitAdapter(256, 200)

