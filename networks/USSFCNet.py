import torch.nn as nn
import torch
from thop import profile

from networks.modules.MSDConv_SSFC import MSDConv_SSFC


class First_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(First_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.Conv = nn.Sequential(
            MSDConv_SSFC(in_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            MSDConv_SSFC(out_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.Conv(input)


class USSFCNet(nn.Module):
    # Ratio 0.5 is the exact paper architecture yielding 1.52M active parameters
    def __init__(self, in_ch, out_ch, ratio=0.5):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(2)
        self.Conv1_1, self.Conv1_2 = First_DoubleConv(in_ch, int(64*ratio)), First_DoubleConv(in_ch, int(64*ratio))
        self.Conv2_1, self.Conv2_2 = DoubleConv(int(64*ratio), int(128*ratio)), DoubleConv(int(64*ratio), int(128*ratio))
        self.Conv3_1, self.Conv3_2 = DoubleConv(int(128*ratio), int(256*ratio)), DoubleConv(int(128*ratio), int(256*ratio))
        self.Conv4_1, self.Conv4_2 = DoubleConv(int(256*ratio), int(512*ratio)), DoubleConv(int(256*ratio), int(512*ratio))
        self.Conv5_1, self.Conv5_2 = DoubleConv(int(512*ratio), int(1024*ratio)), DoubleConv(int(512*ratio), int(1024*ratio))
        
        self.Up5, self.Up_c5 = nn.ConvTranspose2d(int(1024*ratio), int(512*ratio), 2, stride=2), DoubleConv(int(1024*ratio), int(512*ratio))
        self.Up4, self.Up_c4 = nn.ConvTranspose2d(int(512*ratio), int(256*ratio), 2, stride=2), DoubleConv(int(512*ratio), int(256*ratio))
        self.Up3, self.Up_c3 = nn.ConvTranspose2d(int(256*ratio), int(128*ratio), 2, stride=2), DoubleConv(int(256*ratio), int(128*ratio))
        self.Up2, self.Up_c2 = nn.ConvTranspose2d(int(128*ratio), int(64*ratio), 2, stride=2), DoubleConv(int(128*ratio), int(64*ratio))
        self.Out = nn.Conv2d(int(64*ratio), out_ch, 1)

    def forward(self, t1, t2):
        c1_1, c1_2 = self.Conv1_1(t1), self.Conv1_2(t2)
        x1 = torch.abs(c1_1 - c1_2)
        c2_1, c2_2 = self.Conv2_1(self.Maxpool(c1_1)), self.Conv2_2(self.Maxpool(c1_2))
        x2 = torch.abs(c2_1 - c2_2)
        c3_1, c3_2 = self.Conv3_1(self.Maxpool(c2_1)), self.Conv3_2(self.Maxpool(c2_2))
        x3 = torch.abs(c3_1 - c3_2)
        c4_1, c4_2 = self.Conv4_1(self.Maxpool(c3_1)), self.Conv4_2(self.Maxpool(c3_2))
        x4 = torch.abs(c4_1 - c4_2)
        c5_1, c5_2 = self.Conv5_1(self.Maxpool(c4_1)), self.Conv5_2(self.Maxpool(c4_2))
        x5 = torch.abs(c5_1 - c5_2)
        
        d5 = self.Up_c5(torch.cat((x4, self.Up5(x5)), 1))
        d4 = self.Up_c4(torch.cat((x3, self.Up4(d5)), 1))
        d3 = self.Up_c3(torch.cat((x2, self.Up3(d4)), 1))
        d2 = self.Up_c2(torch.cat((x1, self.Up2(d3)), 1))
        return torch.sigmoid(self.Out(d2))