import torch
import torch.nn as nn
from torch import Tensor

class SeBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        C = in_channels
        r = C // 4
        self.globpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(C, r, bias=False)
        self.fc2 = nn.Linear(r, C, bias=False)
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        f = self.globpool(x)
        f = torch.flatten(f, 1)
        f = self.relu(self.fc1(f))
        f = self.hsigmoid(self.fc2(f))
        f = f[:, :, None, None]
        scale = x * f
        return scale
