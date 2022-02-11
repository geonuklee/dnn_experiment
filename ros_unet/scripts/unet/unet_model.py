""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

if __package__ == 'unet':
    from .unet_parts import *
else:
    from unet_parts import *

class DuNet(nn.Module):

    def __init__(self):
        super(DuNet, self).__init__()
        # Refinery module, mini UNet
        self.conv1_d    = DoubleDown(3, 32)
        self.conv2 = DoubleDown(self.conv1_d.out_channnels, 64)
        self.conv3 = DoubleDown(self.conv2.out_channnels, 128)
        self.conv4 = DoubleDown(self.conv3.out_channnels, 256)

        self.up3 = Up(self.conv3.out_channnels+self.conv4.out_channnels, 128, kernel_size=3)
        self.up2 = Up(self.conv2.out_channnels+self.up3.out_channnels, 64, kernel_size=3)

        #self.up1_rgbd = Up(self.conv1_rgbd.out_channnels+self.up2.out_channnels, 3, kernel_size=3)
        self.up1_d    = Up(   self.conv1_d.out_channnels+self.up2.out_channnels, 3, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=1) # c of b-c-h-w

        self.loss_bg_edge_box = nn.CrossEntropyLoss(weight=torch.Tensor([0.01, 1., 1.]) )
        self.loss_bg_edge = nn.CrossEntropyLoss(weight=torch.Tensor([0.01, 1.]) )


    def forward(self, x):
        # TODO Adding iteration
        #if x.shape[1] == 6:
        #    conv1, up1 = self.conv1_rgbd, self.up1_rgbd
        #else:
        #    conv1, up1 = self.conv1_d, self.up1_d

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

