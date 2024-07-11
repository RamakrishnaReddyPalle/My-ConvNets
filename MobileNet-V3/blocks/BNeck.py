import torch
import torch.nn as nn
from torch import Tensor
from blocks.ConvBlock import ConvBlock
from blocks.SeBlock import SeBlock

class BNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        exp_size: int,
        se: bool,
        act: nn.Module,
        stride: int
    ):
        super().__init__()
        self.add = in_channels == out_channels and stride == 1

        self.block = nn.Sequential(
            ConvBlock(in_channels, exp_size, 1, 1, act),
            ConvBlock(exp_size, exp_size, kernel_size, stride, act, exp_size),
            SeBlock(exp_size) if se else nn.Identity(),
            ConvBlock(exp_size, exp_size, kernel_size, 1, act, exp_size),
            ConvBlock(exp_size, out_channels, 1, 1, act=nn.Identity())
        )

    def forward(self, x: Tensor) -> Tensor:
        res = self.block(x)
        if self.add:
            res = res + x
        return res
