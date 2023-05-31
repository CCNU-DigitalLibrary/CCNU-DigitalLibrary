from torch import nn
from lib.models.layers.pooling import *


class BaseHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def _init_weight(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
            elif classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def construct_pool_layer(self, visual_pool_type, textual_pool_type):
        if visual_pool_type == 'fastavgpool':
            self.visual_pool_layer = FastGlobalAvgPool2d()
        elif visual_pool_type == 'avgpool':
            self.visual_pool_layer = nn.AdaptiveAvgPool2d(1)
        elif visual_pool_type == 'maxpool':
            self.visual_pool_layer = nn.AdaptiveMaxPool2d(1)
        elif visual_pool_type == 'gempoolP':
            self.visual_pool_layer = GeneralizedMeanPoolingP()
        elif visual_pool_type == 'gempool':
            self.visual_pool_layer = GeneralizedMeanPooling()
        elif visual_pool_type == "avgmaxpool":
            self.visual_pool_layer = AdaptiveAvgMaxPool2d()
        elif visual_pool_type == 'clipavgpool':
            self.visual_pool_layer = ClipGlobalAvgPool2d()
        elif visual_pool_type == "identity":
            self.visual_pool_layer = nn.Identity()
        elif visual_pool_type == "flatten":
            self.visual_pool_layer = Flatten()
        else:
            raise KeyError(f"{visual_pool_type} is not supported!")

        if textual_pool_type == 'fastavgpool':
            self.textual_pool_layer = FastGlobalAvgPool2d()
        elif textual_pool_type == 'avgpool':
            self.textual_pool_layer = nn.AdaptiveAvgPool2d(1)
        elif textual_pool_type == 'maxpool':
            self.textual_pool_layer = nn.AdaptiveMaxPool2d(1)
        elif textual_pool_type == 'gempoolP':
            self.textual_pool_layer = GeneralizedMeanPoolingP()
        elif textual_pool_type == 'gempool':
            self.textual_pool_layer = GeneralizedMeanPooling()
        elif textual_pool_type == "avgmaxpool":
            self.textual_pool_layer = AdaptiveAvgMaxPool2d()
        elif textual_pool_type == 'clipavgpool':
            self.textual_pool_layer = ClipGlobalAvgPool2d()
        elif textual_pool_type == "identity":
            self.textual_pool_layer = nn.Identity()
        elif textual_pool_type == "flatten":
            self.textual_pool_layer = Flatten()
        else:
            raise KeyError(f"{textual_pool_type} is not supported!")

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=stride,
                     padding=(0, 1), groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.linear = nn.Linear(512, 512)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)




        return out

# class Bottleneck_with_rrelu(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  width=64, dilation=1, norm_layer=None):
#         super(Bottleneck_with_rrelu, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv1x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes)
#         self.bn3 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.rrelu =nn.RReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.rrelu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.rrelu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.rrelu(out)
#
#         return out
