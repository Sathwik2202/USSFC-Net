import torch
import torch.nn as nn
import numpy as np


class SSFC(nn.Module):
    def __init__(self, in_ch):
        super(SSFC, self).__init__()
    def forward(self, x):
        _, _, h, w = x.size()
        q = x.mean(dim=[2, 3], keepdim=True)
        square = (x - q).pow(2)
        sigma = square.sum(dim=[2, 3], keepdim=True) / (h * w)
        att_weight = torch.sigmoid(square / (2 * sigma + np.finfo(np.float32).eps) + 0.5)
        return x * att_weight