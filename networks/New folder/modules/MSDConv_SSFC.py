import math
import torch
from torch import nn

from networks.modules.SSFC import SSFC
from networks.modules.CMConv import CMConv


class MSDConv_SSFC(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=2, dilation=3):
        super(MSDConv_SSFC, self).__init__()
        self.out_ch = out_ch
        n_ch = math.ceil(out_ch / ratio)
        a_ch = n_ch * (ratio - 1)
        self.native = nn.Sequential(nn.Conv2d(in_ch, n_ch, 1, bias=False), nn.BatchNorm2d(n_ch), nn.ReLU(True))
        self.aux = nn.Sequential(CMConv(n_ch, a_ch, groups=int(n_ch/4), dilation=dilation), nn.BatchNorm2d(a_ch), nn.ReLU(True))
        self.att = SSFC(a_ch)
    def forward(self, x):
        x1 = self.native(x)
        x2 = self.att(self.aux(x1))
        return torch.cat([x1, x2], dim=1)[:, :self.out_ch, :, :]