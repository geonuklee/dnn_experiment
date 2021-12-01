""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

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
        self.in_channels = 1
        n_cls = 2

        # Refinery module, mini UNet
        self.conv1 = DoubleDown(self.in_channels, 32, kernel_size=3)
        self.conv2 = DoubleDown(self.conv1.out_channnels, 32)
        self.conv3 = DoubleDown(self.conv2.out_channnels, 64)
        self.conv4 = DoubleDown(self.conv3.out_channnels, 256)

        self.up3 = Up(self.conv3.out_channnels+self.conv4.out_channnels, 128, kernel_size=3)
        self.up2 = Up(self.conv2.out_channnels+self.up3.out_channnels, 64, kernel_size=3)
        self.up1 = Up(self.conv1.out_channnels+self.up2.out_channnels, 32, kernel_size=3)
        self.up0 = Up(1                       +self.up1.out_channnels, 2, kernel_size=3)

        self.softmax = nn.Softmax(dim=1) # c of b-c-h-w
        weights = torch.tensor( [0.1, 2.]) # Improved result with weight.
        self.f_loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, x):
        # TODO Adding iteration
        l0 = torch.unsqueeze(x[:,0,:,:], 1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = self.up3(x4, x3)
        x = self.up2(x,  x2)
        x = self.up1(x,  x1)
        x = self.up0(x,  l0)
        x = self.softmax(x)
        return x

    def loss(self, pred, target):
        return self.f_loss(pred, target)


class ShallowCNN(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        n_cls = 2 # non-groove 0, groove 1
        self.bn = nn.BatchNorm2d(n_channels)
        n = 16
        self.bn2 = nn.BatchNorm2d(n)
        self.conv1 = nn.Conv2d(n_channels, n, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(n, n_cls, kernel_size=5,padding=2)
        self.softmax = nn.Softmax(dim=1) # c of b-c-h-w

        self.f_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.softmax(x)
        return x

    def loss(self, pred, target):
        return self.f_loss(pred, target)

