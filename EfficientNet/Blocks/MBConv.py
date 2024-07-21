import torch
import torch.nn as nn
from stochastic_depth import StochasticDepth


# MBConv
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, exp, r):
        super().__init__()
        exp_channels = in_channels * exp
        self.add = in_channels == out_channels and stride == 1
        self.c1 = ConvBlock(in_channels, exp_channels, 1, 1) if exp > 1 else nn.Identity()
        self.c2 = ConvBlock(exp_channels, exp_channels, kernel_size, stride, exp_channels)
        self.se = SeBlock(exp_channels, r)
        self.c3 = ConvBlock(exp_channels, out_channels, 1, 1, act=False)

        " Stochastic Depth module with default survival probability 0.5. "
        self.sd = StochasticDepth()

    def forward(self, x):
        f = self.c1(x)
        f = self.c2(f)
        f = self.se(f)
        f = self.c3(f)

        if self.add:
            f = x + f

        f = self.sd(f)

        return f