""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, kernel_size=3, stride=1):
        super(DoubleConv,self).__init__()
        if stride == 1:
            pad = int( (kernel_size-1)/2 )
        else:
            pad = 0
        self.seq = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=kernel_size, padding=pad),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, kernel_size=kernel_size, padding=pad),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
                )
        self.out_channnels = out_ch

    def forward(self, x):
        return self.seq(x)


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SingleConv,self).__init__()
        pad = int( (kernel_size-1)/2 )
        self.seq = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                )
        self.out_channnels = out_ch

    def forward(self, x):
        return self.seq(x)


class DoubleDown(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(DoubleDown,self).__init__()
        pad = int( (kernel_size-1)/2 )
        self.seq =  nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_ch, out_ch, out_ch, kernel_size=kernel_size)
                )
        self.out_channnels = out_ch

    def forward(self, x):
        return self.seq(x)

class SingleDown(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SingleDown,self).__init__()
        self.seq =  nn.Sequential(
                nn.MaxPool2d(2),
                SingleConv(in_ch, out_ch, kernel_size=kernel_size)
                )
        self.out_channnels = out_ch

    def forward(self, x):
        return self.seq(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch=in_ch, mid_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size)

        self.out_channnels = out_ch

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SimpleUp(nn.Module):
    def __init__(self, scale_factor=2):
        super(SimpleUp, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x1, w,h):
        x1 = self.up(x1)
        diffY = h - x1.size()[2]
        diffX = w - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x1
