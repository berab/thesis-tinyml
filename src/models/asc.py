import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import torchaudio.transforms as tf


class SubSpectralNorm(nn.Module):
    def __init__(self, C, S):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.bn = nn.BatchNorm2d(C*S)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)
        x = self.bn(x)
        return x.view(N, C, F, T)


class BroadcastedBlock(nn.Module):
    def __init__(
            self,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(BroadcastedBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation,
                                      stride=stride, bias=True)
        self.ssn1 = SubSpectralNorm(planes, 4)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.channel_drop = nn.Dropout2d(p=0.1)
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)

        return out


class TransitionBlock(nn.Module):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(TransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride,
                                      dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, 4)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.channel_drop = nn.Dropout2d(p=0.1)
        self.swish = nn.SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        # f2
        #############################
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ssn(out)
        out = self.freq_dw_conv(out)
        #############################
        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        #############################

        out = auxilary + out
        out = self.relu(out)

        return out


class BCResNetMod1(torch.nn.Module):
    def __init__(self, n_class, n_chan, quant=None):
        super(BCResNetMod1, self).__init__()
        self.quant = quant
        self.conv1 = nn.Conv2d(n_chan, 20, 5, stride=2, padding=2, bias=False)
        self.block1_1 = TransitionBlock(20, 10)
        self.block1_2 = BroadcastedBlock(10)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.block2_1 = TransitionBlock(10, 15)
        self.block2_2 = BroadcastedBlock(15)

        self.block3_1 = TransitionBlock(15, 20)
        self.block3_2 = BroadcastedBlock(20)

        self.block4_1 = TransitionBlock(20, 25)
        self.block4_2 = BroadcastedBlock(25)
        self.block4_3 = BroadcastedBlock(25)

        self.conv2 = nn.Conv2d(25, n_class, 1, bias=False)

    def forward(self, x):

        out = self.conv1(x)
        
        out = self.block1_1(out)
        out = self.block1_2(out)
        out = self.maxpool(out)

        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.maxpool(out)

        out = self.block3_1(out)
        out = self.block3_2(out)

        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)

        out = self.conv2(out)
        out = out.mean(dim=(2,3))

        return out


class BCResNetMod8(torch.nn.Module):
    def __init__(self, n_chan, n_class, quant=None):
        super(BCResNetMod8, self).__init__()
        self.quant = quant
        self.conv1 = nn.Conv2d(n_chan, 160, 5, stride=2, padding=2, bias=False)
        self.block1_1 = TransitionBlock(160, 80)
        self.block1_2 = BroadcastedBlock(80)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.block2_1 = TransitionBlock(80, 120)
        self.block2_2 = BroadcastedBlock(120)

        self.block3_1 = TransitionBlock(120, 160)
        self.block3_2 = BroadcastedBlock(160)

        self.block4_1 = TransitionBlock(160, 200)
        self.block4_2 = BroadcastedBlock(200)
        self.block4_3 = BroadcastedBlock(200)

        self.conv2 = nn.Conv2d(200, n_class, 1, bias=False)

    def forward(self, x):

        out = self.conv1(x)
        
        out = self.block1_1(out)
        out = self.block1_2(out)
        out = self.maxpool(out)

        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.maxpool(out)

        out = self.block3_1(out)
        out = self.block3_2(out)

        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)

        out = self.conv2(out)
        out = out.mean(dim=(2,3))

        return out


# --- Test ---
def test_bcresnetmod1():
    x = torch.ones(4, 1, 256, 334)
    net = BCResNetMod1(n_class=10, n_chan=1)
    output = net(x)
    n_params =  sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert output.shape == (4, 10)
    assert n_params == 8100

def test_bcresnetmod1():
    x = torch.ones(4, 1, 256, 334)
    net = BCResNetMod8(n_class=10, n_chan=1)
    output = net(x)
    n_params =  sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert output.shape == (4, 10)
    assert n_params == 315400

