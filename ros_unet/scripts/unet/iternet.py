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
        self.co = conv1L(dim_in= 32, dim_out=  1, kernel=1, stride=1, padding=0, bias=True)
        
    def forward(self, x0, x6, x9s=None):
        x9 = self.c0(x6)
        if x9s is not None:
            x9_ = torch.cat((x9s, x9), dim=1)
            x9_ = torch.cat((x9_, x0), dim=1)
            x9s = torch.cat((x9, x9s), dim=1)
        else:
            x9s = x9
            x9_ = torch.cat((x9, x0), dim=1)
        x0c = self.cd(x9_)
        x1p = self.pool(x0c)
        x1c = self.c1(x1p)
        x2p = self.pool(x1c)
        x2c = self.c2(x2p)
        x3p = self.pool(x2c)
        x3 = self.c3(x3p)
        x3 = torch.cat((x3, x2c), dim=1)
        x4 = self.c4(x3)
        x4 = torch.cat((x4, x1c), dim=1)
        x5 = self.c5(x4)
        x5 = torch.cat((x5, x0c), dim=1)
        x6 = self.c6(x5)
        xout = self.co(x6)
        return xout, x6, x9s
        
class IterNetInit(nn.Module):
    def __init__(self):
        super(IterNetInit, self).__init__()
        self.block1_pool = maxPool()
        self.block1_c1 = conv2L(dim_in=  4, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c2 = conv2L(dim_in= 32, dim_out= 64, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c3 = conv2L(dim_in= 64, dim_out=128, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c4 = conv2L(dim_in=128, dim_out=256, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c5 = convUp(dim_in=256, dim_out=256, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c6 = convUp(dim_in=512, dim_out=128, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c7 = convUp(dim_in=256, dim_out= 64, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c8 = convUp(dim_in=128, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c9 = conv2L(dim_in= 64, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.block1_co = conv1L(dim_in= 32, dim_out=  1, kernel=1, stride=1, padding=0, bias=True)
        
    def forward(self, img):
        x1c = self.block1_c1(img)
        x1p = self.block1_pool(x1c)
        x2c = self.block1_c2(x1p)
        x2p = self.block1_pool(x2c)
        x3c = self.block1_c3(x2p)
        x3p = self.block1_pool(x3c)
        x4c = self.block1_c4(x3p)
        x4p = self.block1_pool(x4c)
        
        x5 = self.block1_c5(x4p)
        x5 = torch.cat((x5, x4c), dim=1)
        x6 = self.block1_c6(x5)
        x6 = torch.cat((x6, x3c), dim=1)
        x7 = self.block1_c7(x6)
        x7 = torch.cat((x7, x2c), dim=1)
        x8 = self.block1_c8(x7)
        x8 = torch.cat((x8, x1c), dim=1)
        x9 = self.block1_c9(x8)
        xout = self.block1_co(x9)
        
        return xout, x1c, x9
    

class IterNet(nn.Module):
    def __init__(self):
        super(IterNet, self).__init__()
        self.net1 = IterNetInit()
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

def distance_weighted_bce_loss(spliter, output, target, fn_w, fp_w):
    assert(output.shape[0] == target.shape[0])
    assert(output.shape[1] == 1)
    assert(target.shape[1] == 1)

    dist_weights = torch.zeros_like(output) # weights for false positive
    offset = int( (spliter.wh - spliter.step)/2 )
    nb = output.shape[0]
    for b in range(nb):
        outline = (target[b,0,:,:].numpy() > .5)
        dist = cv2.distanceTransform( (~outline).astype(np.uint8),
                                distanceType=cv2.DIST_L2, maskSize=5)
        w = np.ones_like(dist)
        w[dist < 10.] = .1
        w[dist < 5.] = .01
        dist_weights[b,0,:,:] = torch.Tensor(w).to(dist_weights.device)

    if output.device != torch.device('cpu'):
        #output = output.to('cpu') # For debugging
        #dist_weights = dist_weights.to('cpu')
        target = target.to(output.device)
    zeros = torch.zeros_like(output)

    yn = target*output
    yp = dist_weights*output
    fn_loss = fn_w * f.binary_cross_entropy( yn, target)
    fp_loss = fp_w * f.binary_cross_entropy( yp, target)
    return fn_loss+fp_loss

def masked_loss(spliter, output, target, validmask, outline_dist, fn_w, fp_w):
    assert(output.shape[0] == target.shape[0])
    assert(output.shape[1] == 1)
    assert(target.shape[1] == 1)

    offset = 26

    if output.device != torch.device('cpu'):
        target = target.to(output.device)

    cropped_output = output[:,:,offset:-offset-1, offset:-offset-1]
    cropped_target = target[:,:,offset:-offset-1, offset:-offset-1]
    cropped_weights = torch.ones_like(cropped_target)
    cropped_weights[validmask[:,:,offset:-offset-1, offset:-offset-1]==0] = 0.
    cropped_weights = cropped_weights.detach()
    yn = cropped_target*cropped_output
    fn_loss  = fn_w * F.binary_cross_entropy( yn,             cropped_target, cropped_weights)
    fnp_loss = fp_w * F.binary_cross_entropy( cropped_output, cropped_target, cropped_weights)
    return fn_loss+fnp_loss

def masked_loss2(spliter, output, target, validmask, outline_dist, fn_w, fp_w):
    assert(output.shape[1] == 1)
    assert(target.shape[1] == 1)

    l = 20
    offset = int(l/2)

    if output.device != torch.device('cpu'):
        target = target.to(output.device)

    ### step 1. outline weight
    cropped_output = output[:,:,offset:-offset-1, offset:-offset-1]
    cropped_target = target[:,:,offset:-offset-1, offset:-offset-1]

    #alpha = .2
    #t1 = cropped_target * F.conv2d(target-target_output, ones)
    #t2 = F.conv2d(target, ones)
    #t2[cropped_target == 0.] = 1.
    #outline_weights = t1 /t2
    outline_weights = outline_weights* (1.-alpha) + alpha # weight : alpha ~ 2.

    #### step 2. non outline weights
    alpha = .2
    non_outline_weights = outline_dist[:,:,offset:-offset-1, offset:-offset-1]/50.
    non_outline_weights[non_outline_weights > 1.] = 1.
    non_outline_weights = non_outline_weights* (2.-alpha) + alpha
    non_outline_weights = non_outline_weights.to(output.device)

    cropped_weights = torch.ones_like(cropped_target)
    b = outline_dist < 5.
    #cropped_weights[b] =     outline_weights[b] # Make worse
    cropped_weights[~b] = non_outline_weights[~b]
    cropped_weights[validmask[:,:,offset:-offset-1, offset:-offset-1]==0] = 0.
    #cropped_weights = cropped_weights.detach()

    yn = cropped_target*cropped_output
    fn_loss  = 1e+3 * fn_w * F.binary_cross_entropy( yn,      cropped_target, cropped_weights)
    fnp_loss = fp_w * F.binary_cross_entropy( cropped_output, cropped_target, cropped_weights)
    return fn_loss+fnp_loss

