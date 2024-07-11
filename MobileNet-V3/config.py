import torch.nn as nn

def get_config(name):
    HE, RE = nn.Hardswish(), nn.ReLU()
    # [kernel, exp_size, in_channels, out_channels, SEBlock(SE), activation function(NL), stride(s)] 
    large = [
            [3, 16, 16, 16, False, RE, 1],
            [3, 64, 16, 24, False, RE, 2],
            [3, 72, 24, 24, False, RE, 1],
            [5, 72, 24, 40, True, RE, 2],
            [5, 120, 40, 40, True, RE, 1],
            [5, 120, 40, 40, True, RE, 1],
            [3, 240, 40, 80, False, HE, 2],
            [3, 200, 80, 80, False, HE, 1],
            [3, 184, 80, 80, False, HE, 1],
            [3, 184, 80, 80, False, HE, 1],
            [3, 480, 80, 112, True, HE, 1],
            [3, 672, 112, 112, True, HE, 1],
            [5, 672, 112, 160, True, HE, 2],
            [5, 960, 160, 160, True, HE, 1],
            [5, 960, 160, 160, True, HE, 1]
    ]

    small = [
            [3, 16, 16, 16, True, RE, 2],
            [3, 72, 16, 24, False, RE, 2],
            [3, 88, 24, 24, False, RE, 1],
            [5, 96, 24, 40, True, HE, 2],
            [5, 240, 40, 40, True, HE, 1],
            [5, 240, 40, 40, True, HE, 1],
            [5, 120, 40, 48, True, HE, 1],
            [5, 144, 48, 48, True, HE, 1],
            [5, 288, 48, 96, True, HE, 2],
            [5, 576, 96, 96, True, HE, 1],
            [5, 576, 96, 96, True, HE, 1]
    ]

    if name == "large": 
        return large
    if name == "small": 
        return small
    raise ValueError("Invalid configuration name. Choose 'large' or 'small'.")
