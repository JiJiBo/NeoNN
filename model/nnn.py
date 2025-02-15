import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class NNNModel(nn.Module):
    def __init__(self, number_class=10):
        super(NNNModel, self).__init__()
        self.number_class = number_class
        self.c1 = ConvBlock(in_channel=1, out_channel=64, kernel_size=3, stride=1, padding=1)
        self.c2 = ConvBlock(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1)
        self.c3 = ConvBlock(in_channel=64, out_channel=128, kernel_size=3, stride=1, padding=1)
        self.c4 = ConvBlock(in_channel=192, out_channel=256, kernel_size=3, stride=1, padding=1)
        self.c5 = ConvBlock(in_channel=192, out_channel=256, kernel_size=3, stride=1, padding=1)
        self.c6 = ConvBlock(in_channel=640, out_channel=256, kernel_size=3, stride=1, padding=1)
        self.c7 = ConvBlock(in_channel=256, out_channel=128, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc0 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, number_class)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2 + c1)
        c4 = self.c4(torch.cat([c3, c1], dim=1))
        c5 = self.c5(torch.cat([c3, c2], dim=1))
        c6 = self.c6(torch.cat([c3, c4, c5], dim=1))
        c7 = self.c7(c6 + c4 + c5)
        out = self.dropout(c7)
        out = self.fc0(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
