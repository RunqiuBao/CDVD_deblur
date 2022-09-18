import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from .arches import conv1x1, conv3x3, conv5x5, actFunc

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y, y

# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='relu', do_norm=False):
        super(dense_layer, self).__init__()
        self.do_norm = do_norm
        self.conv = conv3x3(in_channels, growthRate)
        if self.do_norm==True:
            self.norm = nn.BatchNorm2d(growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        if self.do_norm==True:
            out = self.act(self.norm(self.conv(x)))
        else:
            out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu', do_norm=False, do_attention=False):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        self.do_attention = do_attention
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation, do_norm=do_norm))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)
        if self.do_attention:
            self.ca = CALayer(in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        if self.do_attention:
            out,_ = self.ca(out)
        out += x
        return out


# Middle network of residual dense blocks
class RDNet(nn.Module):
    def __init__(self, in_channels, out_channels, growthRate, num_layer, num_blocks, activation='relu', do_norm=False, do_attention=True):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer, activation, do_norm=do_norm, do_attention=do_attention))
        self.conv1x1 = conv1x1(num_blocks * in_channels, out_channels)
        self.conv3x3 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out

class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.device = torch.device('cuda')
        self.efe = RDNet(in_channels=43, out_channels=64, growthRate=32, num_layer=3, num_blocks=9, do_attention=True)
        self.hdproject1_sbt = conv1x1(43, 64)
        self.hdproject2_sbt = conv1x1(64, 64)
        self.hdproject1_logdiff = conv1x1(43, 64)
        self.hdproject2_logdiff = conv1x1(64, 64)
        for param in self.hdproject1_logdiff.parameters():
            param.requires_grad = False
            print("logdiff param.requires_grad1: {}".format(param.requires_grad))
        for param in self.hdproject2_logdiff.parameters():
            param.requires_grad = False
            print("logdiff param.requires_grad2: {}".format(param.requires_grad))
        

    def forward(self, x):
#         print("sbt shape before squeeze: {}".format(x[:,:,0:43,:,:].shape))
        sbt = x[:,:,0:43,:,:].squeeze(1)
        logdiff = x[:,:,43:86,:,:].squeeze(1)
#         print("sbt shape after squeeze: {}".format(sbt.shape))
        sbt_out1 = self.efe(sbt)
        sbt_out2 = self.hdproject1_sbt(sbt)
        sbt_out2 = self.hdproject2_sbt(sbt_out2)
        sbt_out = torch.add(sbt_out1, sbt_out2)
        logdiff_out = self.hdproject1_logdiff(logdiff)
        logdiff_out = self.hdproject2_logdiff(logdiff_out)
        return logdiff_out, sbt_out

    # For calculating GMACs
    def profile_forward(self, x):
        sbt = x[:,:,0:43,:,:].squeeze()
        logdiff = x[:,:,43:86,:,:].squeeze()
        sbt_out1 = self.efe(sbt)
        sbt_out2 = self.hdproject1_sbt(sbt)
        sbt_out2 = self.hdproject2_sbt(sbt_out2)
        sbt_out = torch.add(sbt_out1, sbt_out2)
        logdiff_out = self.hdproject1_logdiff(logdiff)
        logdiff_out = self.hdproject2_logdiff(logdiff_out)
        return logdiff_out, sbt_out


def feed(model, iter_samples):
    sbt = iter_samples[0]
    logdiff = iter_samples[1]
#     print(" sbt shape: {}".format((sbt).shape))
#     print(" logdiff shape: {}".format((logdiff).shape))
    x = torch.cat([sbt, logdiff], dim=2)
    logdiff_out, sbt_out = model(x)
    return logdiff_out, sbt_out


def cost_profile(model, H, W, seq_length):
    sbt = torch.randn(1, seq_length, 43, H, W).cuda()
    logdiff = torch.randn(1, seq_length, 43, H, W).cuda()
    x = torch.cat([sbt,logdiff], dim=2)
#     x = [sbt,logdiff]
    profile_flag = True
    flops, params = profile(model, inputs=(x,), verbose=False)

    return flops / seq_length, params
