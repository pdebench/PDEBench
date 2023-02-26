import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from .fourier_neural_operator import SpectralConv2d_fast as fourier_conv


class FourierConv2d(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 mode1,
                 mode2,
                 padding=0,
                 pad_mode='circular'
                 ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mode1 = mode1
        self.mode2 = mode2
        self.padding = padding
        self.pad_mode = pad_mode

        self.scale = (1 / (in_planes * out_planes))

        self.f_conv = fourier_conv(in_planes, out_planes, mode1, mode2)

    def forward(self, x):
        # x: [b, c, h, w]

        batch_size, in_planes, height, width = x.size()
        if self.padding != 0:
            assert self.padding > 0
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode=self.pad_mode)

        output = self.f_conv(x)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class PeriodicConv2d(nn.Module):
    """Wrapper for Conv2d with periodic padding"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 pad=1,
                 bias=False):
        super().__init__()
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            self.stride = (stride, stride)
        self.filters = nn.Parameter(torch.randn(out_channels, in_channels,
                                           kernel_size[0], kernel_size[1]))
        self.pad = pad
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels,))
        else:
            self.bias = None

    def forward(self, x):
        x = F.pad(x, pad=(self.pad, self.pad, self.pad, self.pad), mode='circular')
        if self.bias is not None:
            x = F.conv2d(x, weight=self.filters, bias=self.bias, stride=self.stride)
        else:
            x = F.conv2d(x, weight=self.filters, stride=self.stride)
        return x


class PeriodicConv3d(nn.Module):
    """Wrapper for Conv3d with periodic padding, the periodic padding only happens in the temporal dimension"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 spatial_pad=1,
                 temp_pad=1,
                 pad_mode='constant',      # this pad mode is for temporal padding
                 bias=False):
        super().__init__()
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        assert len(kernel_size) == 3
        if not isinstance(stride, tuple):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride
        assert len(stride) == 3
        self.filters = nn.Parameter(torch.randn(out_channels, in_channels,
                                                kernel_size[0], kernel_size[1], kernel_size[2]))
        self.spatial_pad = spatial_pad
        self.temp_pad = temp_pad
        self.pad_mode = pad_mode
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels,))
        else:
            self.bias = None

    def forward(self, x):
        # only pad spatial dimension with PBC
        x = F.pad(x, pad=(self.spatial_pad, self.spatial_pad, self.spatial_pad, self.spatial_pad, 0, 0), mode='circular')
        # now pad time dimension
        x = F.pad(x, pad=(0, 0, 0, 0, self.temp_pad, self.temp_pad), mode=self.pad_mode)

        if self.bias is not None:
            x = F.conv3d(x, weight=self.filters, bias=self.bias, stride=self.stride)
        else:
            x = F.conv3d(x, weight=self.filters, bias=None, stride=self.stride)
        return x


def UpBlock(in_planes, out_planes):
    """Simple upsampling block"""
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False, padding_mode='circular'),
        nn.InstanceNorm2d(out_planes * 2),
        nn.GLU(dim=1),
        nn.Conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False, padding_mode='circular'),
        nn.InstanceNorm2d(out_planes * 2),
        nn.GLU(dim=1),
    )

    return block
