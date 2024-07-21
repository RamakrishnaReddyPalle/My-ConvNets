import torch
import torch.nn as nn
from stochastic_depth import StochasticDepth

# Squeeze-and-Excitation Block
class SeBlock(nn.Module):
    def __init__(self, in_channels, r):
        super().__init__()
        C = in_channels
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(C, C//r, bias=False)
        self.fc2 = nn.Linear(C//r, C, bias=False)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """ x shape: [N, C, H, W]. """ 
        f = self.globpool(x)
        f = torch.flatten(f,1)
        f = self.silu(self.fc1(f))
        f = self.sigmoid(self.fc2(f))
        f = f[:,:,None,None]
        """ f shape: [N, C, 1, 1] """ 

        scale = x * f
        return scale