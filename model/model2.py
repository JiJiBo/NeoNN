import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class NNNConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(NNNConvBlock, self).__init__()
        _channel = out_channel // 4
        self.c1 = nn.Conv2d(in_channel, _channel, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(_channel, _channel, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(_channel, _channel, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(_channel, _channel, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(_channel * 4, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c1 + c2)
        c4 = self.c4(c1 + c2 + c3)
        c5 = self.c5(torch.cat([c1, c2, c3, c4], dim=1))
        return c5


class DiyModel2(nn.Module):
    def __init__(self, number_class=10):
        super(DiyModel2, self).__init__()
        self.number_class = number_class
        self.c1 = NNNConvBlock(in_channel=1, out_channel=64)
        self.c2 = NNNConvBlock(in_channel=64, out_channel=128)
        self.c3 = NNNConvBlock(in_channel=128, out_channel=256)
        self.c4 = NNNConvBlock(in_channel=256, out_channel=256)
        self.c5 = NNNConvBlock(in_channel=256, out_channel=128)
        self.c6 = NNNConvBlock(in_channel=128, out_channel=64)
        self.c7 = NNNConvBlock(in_channel=64, out_channel=64)
        self.dropout = nn.Dropout(0.5)
        self.fc0 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, number_class)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        c6 = self.c6(c5)
        c7 = self.c7(c6)
        out = self.dropout(c7)
        out = self.fc0(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out