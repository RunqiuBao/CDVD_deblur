import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from .arches import conv1x1, conv3x3, conv5x5, actFunc

'''from here, IGP'''

do_norm=False

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

class activation(nn.Module):
    def __init__(self, n_ch, do_norm=False):
        super(activation, self).__init__()
        self.do_norm = do_norm
        if self.do_norm == True:
            self.norm = nn.BatchNorm2d(n_ch)
        self.act = nn.GELU()
        
    def forward(self, x):
        if self.do_norm == True:
            out = self.norm(x)
        else:
            out = x           
        out = self.act(out)
        return out

class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, kernerl_size=3, do_norm=False):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.activation = activation(growthRate, do_norm=do_norm)
    def forward(self, x):
        out = self.activation(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, do_norm=False, do_attention=False):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        self.do_attention = do_attention
            
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, do_norm=do_norm))
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


# network of residual dense blocks
class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks, do_norm=False, do_attention=False):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer, do_norm=do_norm,do_attention=do_attention))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

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


# RDB DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, out_channels, num_layer, do_norm):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, do_norm)
        self.down_sampling = conv5x5(in_channels, out_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out

# caoncatenation fusion
class CAT(nn.Module):
    def __init__(self,para):
        super(CAT,self).__init__()
        self.frames = para.past_frames + para.future_frames + 1
        self.n_feats = para.n_features
        self.center = para.past_frames
        self.fusion = conv1x1(self.frames * (5 * self.n_feats), (5 * self.n_feats))
    def forward(self, hs):
        out = torch.cat(hs, dim=1)
        out = self.fusion(out)
        return out

class RNNCell(nn.Module):
    def __init__(self, para):
        super(RNNCell, self).__init__()
        self.para=para
        self.n_feats = para.n_features  # suppose n_feats = 16
        self.n_blocks = para.n_blocks_a  # supporse n_blocks = 4
        self.F_B0 = conv5x5(48, 32, stride=1)
        self.F_B1 = RDB_DS(in_channels=32, growthRate=self.n_feats, out_channels = 32, num_layer=3, do_norm=do_norm)  # in: 16
        self.F_B2 = RDB_DS(in_channels=32, growthRate=int(self.n_feats * 3 / 2), 
                            out_channels = 32, num_layer=3, do_norm=do_norm)  # in: 32

        self.F_R = RDNet(in_channels=(1+2+2) * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                            num_blocks=self.n_blocks, do_norm=do_norm,do_attention=True)  # in: 80
        # F_h: hidden state part
        self.F_h = nn.Sequential(
            conv3x3(( 1+2+2) * self.n_feats, self.n_feats),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, do_norm=do_norm),
            conv3x3(self.n_feats, self.n_feats)
        )

    def forward(self, input, s_last, mid_last):
        x=input
        #weight = torch.mean(y,dim=-1)
        #weight = torch.mean(weight,dim=-1)
        #weight = weight.unsqueeze(-1).unsqueeze(-1)
        #weight = 1.0-weight

        out = self.F_B0(x)
        out = self.F_B1(out)
        #h = h*weight

        out = self.F_B2(out)
        mid = out
        out = torch.cat([out, s_last,mid_last], dim=1)
        #out = torch.cat([out, s_last], dim=1)
        #h = h*weight

        out = self.F_R(out)
        #out = out*weight
        s = self.F_h(out)
        #out = out*weight

        return out, s, mid

# Reconstructor
class Reconstructor(nn.Module):
    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.para = para
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks_b
        self.model = nn.Sequential(
            RDNet(in_channels= 5 * self.n_feats, growthRate=2 * self.n_feats, num_layer=3, num_blocks=self.n_blocks, do_norm=do_norm, do_attention=True),
            nn.ConvTranspose2d((5 * self.n_feats) , 3 * self.n_feats, kernel_size=3, stride=2,
                                padding=1, output_padding=1),
            nn.ConvTranspose2d(3 * self.n_feats, 24, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv5x5(24, 24, stride=1)
        )

    def forward(self, x):
        return self.model(x)

class IGP3(nn.Module):
    def __init__(self, para):
        super(IGP3, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.ds_ratio = 4
        self.device = torch.device('cuda')
        self.rnncell0 = RNNCell(para)
        self.recons = Reconstructor(para)
        self.ta_fusion = CAT(para)

    def forward(self, x):#n,14,48,h,w
        outputs, hs = [], []
        batch_size, frames, channels, height, width = x.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        # forward h structure: (batch_size, channel, height, width)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        mid = torch.zeros(batch_size, 2*self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            h, s, mid = self.rnncell0(x[:, i, :,:,:], s, mid)
            hs.append(h)
        for i in range(self.num_fb, frames - self.num_ff):
            out = self.ta_fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            outputs.append(out.unsqueeze(dim=1))
        outputs = torch.cat(outputs, dim=1)
#         print('outputs shape', outputs.shape)

        return outputs #n,10,24,h,w