import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  conv2d(channels, channels // reduction, 4, 1, 0, bias=False),
                                  Swish(),
                                  conv2d(channels // reduction, channels, 1, 1, 0, bias=False),
                                  nn.Sigmoid())

    def forward(self, x):
        return self.main(x)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


class SVA_module(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[1, 3, 5, 7], stride=1):
        super(SVA_module, self).__init__()
        self.conv_1 = conv2d(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, bias=False)
        self.bn_1 = batchNorm2d(planes//4)
        self.LR_1 = nn.LeakyReLU(0.2)

        self.conv_2 = conv2d(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, bias=False)
        self.bn_2 = batchNorm2d(planes // 4)
        self.LR_2 = nn.LeakyReLU(0.2)

        self.conv_3 = conv2d(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, bias=False)
        self.bn_3 = batchNorm2d(planes // 4)
        self.LR_3 = nn.LeakyReLU(0.2)

        self.conv_4 = conv2d(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, bias=False)
        self.bn_4 = batchNorm2d(planes // 4)
        self.LR_4 = nn.LeakyReLU(0.2)

        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x1 = self.bn_1(x1)
        x1 = self.LR_1(x1)

        x2 = self.conv_2(x)
        x2 = self.bn_2(x2)
        x2 = self.LR_2(x2)

        x3 = self.conv_3(x)
        x3 = self.bn_3(x3)
        x3 = self.LR_3(x3)

        x4 = self.conv_4(x)
        x4 = self.bn_4(x4)
        x4 = self.LR_4(x4)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((out, x_se_weight_fp), 1)
        return out