import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import torch.optim as optim
from model.base_networks import *
from torchvision.transforms import *

def prepare(para, x, logdiffs):
    rgb = 255.0
    if para.normalized:
        x = x / rgb
        #evt_max = torch.max(torch.tensor([torch.abs(torch.max(x[:,3:,:,:])), torch.abs(torch.min(x[:,3:,:,:]))]))
        repre_max = 255
        logdiffs = logdiffs / repre_max

    return x, logdiffs


def prepare_reverse(para, x):
    rgb = 255.0
    if para.normalized:
        x = x * rgb
        repre_max = 255

    return x

def prepare_reverse_distillation(para, x, y):
    maxval = 127.0
    x = x*maxval/torch.max(torch.abs(x))
    y = y*maxval/torch.max(torch.abs(y))

    return x, y


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type=='batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type =='instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu': return nn.ReLU()
    elif act == 'relu6': return nn.ReLU6()
    elif act == 'leakyrelu': return nn.LeakyReLU(0.1)
    elif act == 'prelu': return nn.PReLU()
    elif act == 'rrelu': return nn.RReLU(0.1, 0.3)
    elif act == 'selu': return nn.SELU()
    elif act == 'celu': return nn.CELU()
    elif act == 'elu': return nn.ELU()
    elif act == 'gelu': return nn.GELU()
    elif act == 'tanh': return nn.Tanh()
    else: raise NotImplementedError
        
class activation(nn.Module):
    def __init__(self, n_ch, do_norm=False):
        super(activation, self).__init__()
        self.do_norm = do_norm
        if self.do_norm == True:
            self.norm = nn.BatchNorm2d(n_ch)
        self.act = nn.ReLU()
        
    def forward(self, x):
        if self.do_norm == True:
            out = self.norm(x)
        else:
            out = x           
        out = self.act(out)
        return out


class RB(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', batch_norm=False):
        super(RB, self).__init__()
        op = []
        for i in range(2):
            op.append(conv3x3(in_channels, out_channels))
            if batch_norm:
                op.append(nn.BatchNorm2d(out_channels))
            if i == 0:
                op.append(actFunc(activation))
        self.main_branch = nn.Sequential(*op)

    def forward(self, x):
        out = self.main_branch(x)
        out += x
        # no activate function in official version ifirnn
        # out = F.relu(out)
        return out


# network of residual blocks
class RNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, activation='relu'):
        super(RNet, self).__init__()
        self.num_blocks = num_blocks
        self.RBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RBs.append(RB(in_channels, out_channels, activation, batch_norm=False))

    def forward(self, x):
        h = x
        for i in range(self.num_blocks):
            h = self.RBs[i](h)
        out = h
        return out
    
# network of residual blocks
class RNet_laplacian(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, activation='relu'):
        super(RNet_laplacian, self).__init__()
        self.num_blocks = num_blocks
        self.RBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RBs.append(RB(in_channels, in_channels, activation, batch_norm=True))
        self.conv1x1 = conv1x1(in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, out_channels)

    def forward(self, x):
        h = x
        for i in range(self.num_blocks):
            h = self.RBs[i](h)
        out = h
        out = self.conv1x1(out)
        out = self.conv3x3(out)
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
    
# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, reduction=16, do_norm=False, do_attention=False):
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
            self.ca = CALayer(in_channels, reduction)
        
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

class RRDB(nn.Module):
    def __init__(self, in_channels, growthRate, reduction=16, num_layer=3, num_blocks=3, do_norm=False, do_attention=False):
        super(RRDB, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer, reduction=reduction, do_norm=do_norm, do_attention=do_attention))

    def forward(self, x):
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
        out = h
        out = out + x
        return out

class RDNet_distillation(nn.Module):
    def __init__(self, in_channels, out_channels, growthRate, num_layer, num_blocks, do_norm=False, do_attention=False):
        super(RDNet_distillation, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer, do_norm=do_norm, do_attention=do_attention))
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

def conv_na(in_cha, out_cha, kernel_size, padding, stride=1, dilation=1, groups=1, bias=True, act_type='relu', norm_type=None, mode='CNA'):
    '''
    Conv layer with normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    c = nn.Conv2d(in_cha, out_cha, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias, groups=groups)
    a = actFunc(act_type) if act_type else None
    if mode =='CNA':
        n = norm(norm_type, out_cha) if norm_type else None
        return nn.Sequential(*filter(lambda x:x!=None, [c,n,a]))
    else:
        raise NotImplementedError('conv_na mode [{:s}] is not found!'.format(mode))
    

# RDB DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, out_channels, growthRate, num_layer, activation='relu'):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, out_channels, stride=2)

    def forward(self, x, h=None):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out, h


# RB DownSampling module
class RB_DS(nn.Module):
    def __init__(self, in_channels, activation='relu'):
        super(RB_DS, self).__init__()
        self.rb = RB(in_channels, in_channels, activation)
        self.down_sampling = conv5x5(in_channels, 2 * in_channels, stride=2)

    def forward(self, x, h=None):
        # x: n,c,h,w
        x = self.rb(x)
        out = self.down_sampling(x)

        return out, h

class GSA_sing(nn.Module):
    def __init__(self, para):
        super(GSA_sing, self).__init__()
        
        self.n_feats = para.n_features
        self.center = para.past_frames
        self.F_f = nn.Sequential(
            nn.Linear(64, 2 * 64),
            actFunc(para.activation),
            nn.Linear(2 * 64, 64),
            nn.Sigmoid()
        )
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(64,  2 * 64),
            conv1x1( 2 * 64, 64)
        )
        # condense layer
        self.condense = conv1x1(64, 32)
        # fusion layer
        self.fusion = conv1x1(32, 32)

    def forward(self, hs):
        cor = hs
        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)
        cor = self.F_p(cor)
        cor = self.condense(w * cor)
        out = self.fusion(cor)  # 5c

        return out

# Global Average Pooling based Temporal Attention Fusion
class GSA(nn.Module):
    def __init__(self, para):
        super(GSA, self).__init__()
        
        self.n_feats = para.n_features
        self.center = para.past_frames
        self.F_f = nn.Sequential(
            nn.Linear(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            actFunc(para.activation),
            nn.Linear(4 * (5 * self.n_feats), 2 * (5 * self.n_feats)),
            nn.Sigmoid()
        )
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            conv1x1(4 * (5 * self.n_feats), 2 * (5 * self.n_feats))
        )
        # condense layer
        self.condense = conv1x1(2 * (5 * self.n_feats), 5 * self.n_feats)
        # fusion layer
        self.fusion = conv1x1(5 * (5 * self.n_feats), 5 * (5 * self.n_feats))

    def forward(self, hs):
        # hs: [(n=batch_size,c=80,h=64,w=64), ..., (n,c,h,w)]
        self.nframes = len(hs)
        f_ref = hs[self.center]
        cor_l = []
        for i in range(self.nframes):
            if i != self.center:
                cor = torch.cat([f_ref, hs[i]], dim=1)
                w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
                if len(w.shape) == 1:
                    w = w.unsqueeze(dim=0)
                w = self.F_f(w)
                w = w.reshape(*w.shape, 1, 1)
                cor = self.F_p(cor)
                cor = self.condense(w * cor)
                cor_l.append(cor)
        cor_l.append(f_ref)
        out = self.fusion(torch.cat(cor_l, dim=1))  # 5c

        return out

# Fusion before channel attention
class GSABefore(nn.Module):
    def __init__(self, para):
        super(GSABefore, self).__init__()
        self.n_feats = para.n_features
        self.center = para.past_frames
        self.F_f = nn.Sequential(
            nn.Linear(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            actFunc(para.activation),
            nn.Linear(4 * (5 * self.n_feats), 2 * (5 * self.n_feats)),
            nn.Sigmoid()
        )
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            conv1x1(4 * (5 * self.n_feats), 2 * (5 * self.n_feats))
        )
        # condense layer
        self.condense = conv1x1(2 * (5 * self.n_feats), 5 * self.n_feats)
        # fusion layer
        self.fusion = conv1x1(5 * (5 * self.n_feats), 5 * (5 * self.n_feats))

    def forward(self, hs):
        # hs: [(n=batch_size,c=80,h=64,w=64), ..., (n,c,h,w)]
        self.nframes = len(hs)
        f_ref = hs[self.center]
        cor_l = []
        for i in range(self.nframes):
            if i != self.center:
                cor = torch.cat([f_ref, hs[i]], dim=1)
                cor = self.F_p(cor)
                w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
                if len(w.shape) == 1:
                    w = w.unsqueeze(dim=0)
                w = self.F_f(w)
                w = w.reshape(*w.shape, 1, 1)
#                 cor = self.F_p(cor)
                cor = self.condense(w * cor)
                cor_l.append(cor)
        cor_l.append(f_ref)
        out = self.fusion(torch.cat(cor_l, dim=1))  # 5c

        return out

# cancatenation fusion
class CAT(nn.Module):
    def __init__(self, para):
        super(CAT, self).__init__()
        
    def forward(self, hs):
        out = torch.cat(hs, dim=1)

        return out
    
    
#----------------------------------------------------------------#
def conv3d1x1(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3d3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv3d5x5(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)

class dense_layer3d(nn.Module):
    def __init__(self, in_channels, growthRate, kernerl_size=3, activation='relu'):
        super(dense_layer3d, self).__init__()
        self.conv = conv3d3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Residual dense block
class RDB3d(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB3d, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer3d(in_channels_, growthRate, activation=activation))
            in_channels_ += growthRate
        self.dense_layers3d = nn.Sequential(*modules)
        self.conv3d1x1 = conv3d1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers3d(x)
        out = self.conv3d1x1(out)
        out += x
        return out

# network of residual dense blocks with conv3d
class RDNet3d(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks, activation='relu'):
        super(RDNet3d, self).__init__()
        self.num_blocks = num_blocks
        self.RDB3ds = nn.ModuleList()
        for i in range(num_blocks):
            self.RDB3ds.append(RDB3d(in_channels, growthRate, num_layer, activation))
        self.conv3d1x1 = conv3d1x1(num_blocks * in_channels, in_channels)
        self.conv3d3x3 = conv3d3x3(in_channels, in_channels)

    def forward(self, x):#[B,1,C,H,W]
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDB3ds[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv3d1x1(out)
        out = self.conv3d3x3(out)
        return out #[B,1,C,H,W]
      
#---------------------------------------------#DDBPN
class DDBPN(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(DDBPN, self).__init__()
        
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        
        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpBlock(base_filter, kernel, stride, padding, 6)
        self.down7 = D_DownBlock(base_filter, kernel, stride, padding, 7)
        self.up8 = D_UpBlock(base_filter, kernel, stride, padding, 7)
        self.down8 = D_DownBlock(base_filter, kernel, stride, padding, 8)
        #self.up9 = D_UpBlock(base_filter, kernel, stride, padding, 8)
        #self.down9 = D_DownBlock(base_filter, kernel, stride, padding, 9)
        #self.up10 = D_UpBlock(base_filter, kernel, stride, padding, 9)
        #self.down10 = D_DownBlock(base_filter, kernel, stride, padding, 10)
        #Reconstruction
        self.output_conv = ConvBlock(num_stages*base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)
        
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        
        concat_h = torch.cat((h1, h2),1)
        l = self.down2(concat_h)
        
        concat_l = torch.cat((l1, l),1)
        h = self.up3(concat_l)
        
        concat_h = torch.cat((concat_h, h),1)
        l = self.down3(concat_h)
        
        concat_l = torch.cat((concat_l, l),1)
        h = self.up4(concat_l)
        
        concat_h = torch.cat((concat_h, h),1)
        l = self.down4(concat_h)
        
        concat_l = torch.cat((concat_l, l),1)
        h = self.up5(concat_l)
        
        concat_h = torch.cat((concat_h, h),1)
        l = self.down5(concat_h)
        
        concat_l = torch.cat((concat_l, l),1)
        h = self.up6(concat_l)
        
        concat_h = torch.cat((concat_h, h),1)
        l = self.down6(concat_h)
        
        concat_l = torch.cat((concat_l, l),1)
        h = self.up7(concat_l)
        
        concat_h = torch.cat((concat_h, h),1)
        l = self.down7(concat_h)
        
        concat_l = torch.cat((concat_l, l),1)
        h = self.up8(concat_l)
        
        concat_h = torch.cat((concat_h, h),1)
        l = self.down8(concat_h)
        
        #concat_l = torch.cat((l, concat_l),1)
        #h = self.up9(concat_l)
        
        #concat_h = torch.cat((h, concat_h),1)
        #l = self.down9(concat_h)
        
        #concat_l = torch.cat((l, concat_l),1)
        #h = self.up10(concat_l)
        
        #concat_h = torch.cat((h, concat_h),1)
        #l = self.down10(concat_h)
        
        concat_l = torch.cat((concat_l, l),1)
        x = self.output_conv(concat_l)
        
        return x, concat_l