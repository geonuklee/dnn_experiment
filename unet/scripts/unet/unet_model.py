""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

if __package__ == 'unet':
    from .unet_parts import *
else:
    from unet_parts import *

class EdgeNet(nn.Module):
    def __init__(self):
        super(EdgeNet, self).__init__()
        self.in_channels = 1
        n_cls = 2

        self.n_layer= 3
        # DoubleDown with 32-16-8-8-8 >=< 8-8-8-2
        self.conv1 = SingleDown(self.in_channels, 32, kernel_size=3)
        self.conv2 = SingleDown(self.conv1.out_channnels, 128)
        self.conv3 = SingleDown(self.conv2.out_channnels, 16)

        self.conv4 = SingleDown(self.conv3.out_channnels, 16)
        self.up3 = Up(1+self.conv3.out_channnels+self.conv4.out_channnels, 16, kernel_size=1)
        self.up2 = Up(1+self.conv2.out_channnels+self.up3.out_channnels, 16, kernel_size=1)

        #self.up2 = Up(1+self.conv2.out_channnels+self.conv3.out_channnels, 16, kernel_size=1)
        self.up1 = Up(1+self.conv1.out_channnels+self.up2.out_channnels, 16, kernel_size=1)
        self.up0 = Up(1                        +self.up1.out_channnels, 2, kernel_size=1)

        self.softmax = nn.Softmax(dim=1) # c of b-c-h-w

        #self.f_loss = nn.CrossEntropyLoss()
        weights = torch.tensor( [0.1, 2.]) # Improved result with weight.
        self.f_loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, x):
        l0 = torch.unsqueeze(x[:,0,:,:], 1)
        l1 = -torch.max_pool2d(-l0, kernel_size=3, stride=2, padding=1)
        l2 = -torch.max_pool2d(-l1, kernel_size=3, stride=2, padding=1)

        if self.n_layer == 4:
            l3 = -torch.max_pool2d(-l2, kernel_size=3, stride=2, padding=1)

            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x4 = self.conv4(x3)

            l3 = l3[:,:,:x3.shape[2],:x3.shape[3]]
            l2 = l2[:,:,:x2.shape[2],:x2.shape[3]]
            l1 = l1[:,:,:x1.shape[2],:x1.shape[3]]

            x = self.up3(x4, torch.cat((l3,x3), 1) )
            x = self.up2( x, torch.cat((l2,x2), 1) )
            x = self.up1( x, torch.cat((l1,x1), 1) )
            x = self.up0( x, l0)

        elif self.n_layer==3:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)

            l2 = l2[:,:,:x2.shape[2],:x2.shape[3]]
            l1 = l1[:,:,:x1.shape[2],:x1.shape[3]]

            x = self.up2(x3, torch.cat((l2,x2), 1) )
            x = self.up1(x,  torch.cat((l1,x1), 1) )
            x = self.up0(x,  l0)

        x = self.softmax(x)
        return x

    def loss(self, pred, target):
        return self.f_loss(pred, target)

class IterNet(nn.Module):
    def __init__(self):
        super(IterNet, self).__init__()
        self.in_channels = 4
        n_cls = 3

        # Refinery module, mini UNet
        self.conv1 = DoubleDown(self.in_channels, 32, kernel_size=3)
        self.conv2 = DoubleDown(self.conv1.out_channnels, 32)
        self.conv3 = DoubleDown(self.conv2.out_channnels, 64)
        self.conv4 = DoubleDown(self.conv3.out_channnels, 256)

        self.up3 = Up(self.conv3.out_channnels+self.conv4.out_channnels, 128, kernel_size=3)
        self.up2 = Up(self.conv2.out_channnels+self.up3.out_channnels, 64, kernel_size=3)
        self.up1 = Up(self.conv1.out_channnels+self.up2.out_channnels, n_cls, kernel_size=3)
        #self.up0 = Up(1                       +self.up1.out_channnels, n_cls, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.softmax = nn.Softmax(dim=1) # c of b-c-h-w
        weights = torch.tensor( [1., 1., 1.])
        self.f_loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, x):
        # TODO Adding iteration
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = self.up3(x4, x3)
        x = self.up2(x,  x2)
        x = self.up1(x,  x1)
        x = self.up(x)
        x = self.softmax(x)
        return x

    def loss(self, pred, target):
        return self.f_loss(pred, target)

class DuNet(nn.Module):

    def __init__(self):
        super(DuNet, self).__init__()
        # Refinery module, mini UNet
        self.conv1_rgbd = DoubleDown(6, 32, kernel_size=3)
        self.conv1_d    = DoubleDown(3, self.conv1_rgbd.out_channnels,
                                     kernel_size=3)

        self.conv2 = DoubleDown(self.conv1_rgbd.out_channnels, 32)
        self.conv3 = DoubleDown(self.conv2.out_channnels, 64)
        self.conv4 = DoubleDown(self.conv3.out_channnels, 256)

        self.up3 = Up(self.conv3.out_channnels+self.conv4.out_channnels, 128, kernel_size=3)
        self.up2 = Up(self.conv2.out_channnels+self.up3.out_channnels, 64, kernel_size=3)

        self.up1_rgbd = Up(self.conv1_rgbd.out_channnels+self.up2.out_channnels, 3, kernel_size=3)
        self.up1_d    = Up(   self.conv1_d.out_channnels+self.up2.out_channnels, 2, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=1) # c of b-c-h-w

        self.loss_bg_edge_box = nn.CrossEntropyLoss(weight=torch.Tensor([0.01, 1., 1.]) )
        self.loss_bg_edge = nn.CrossEntropyLoss(weight=torch.Tensor([0.01, 1.]) )


    def forward(self, x):
        # TODO Adding iteration
        if x.shape[1] == 6:
            conv1, up1 = self.conv1_rgbd, self.up1_rgbd
        else:
            conv1, up1 = self.conv1_d, self.up1_d

        x1 = conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = self.up3(x4, x3)
        x = self.up2(x,  x2)
        x = up1(x,  x1)
        x = self.up(x)
        x = self.softmax(x)
        return x

    def loss(self, pred, target):
        if pred.shape[1] == 3:
            f_loss = self.loss_bg_edge_box
        else:
            f_loss = self.loss_bg_edge
        return f_loss(pred, target)

