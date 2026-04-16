import torch
import torch.nn as nn

class CMConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, groups=1, dilation_set=4, bias=False):
        super(CMConv, self).__init__()
        self.prim = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation, groups=groups*dilation_set, bias=bias)
        self.prim_shift = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2*dilation, dilation=2*dilation, groups=groups*dilation_set, bias=bias)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)
        
        self.mask = torch.zeros(self.conv.weight.shape).bool().to(device)
        _in, _out = in_ch // (groups * dilation_set), out_ch // (groups * dilation_set)
        for i in range(dilation_set):
            for j in range(groups):
                self.mask[(i+j*groups)*_out : (i+j*groups+1)*_out, i*_in : (i+1)*_in, :, :] = True
        
        self.conv.weight.data[self.mask] = 0
        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        return self.prim(x) + self.conv(x) + self.prim_shift(x_merge)
