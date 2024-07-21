import torch
import torch.nn as nn
from stochastic_depth import StochasticDepth

# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, act=True, bias=False):
        super().__init__()
        """ If k = 1 -> p = 0, k = 3 -> p = 1, k = 5, p = 2. """
        padding = kernel_size // 2
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.silu(self.bn(self.c(x)))
        return x