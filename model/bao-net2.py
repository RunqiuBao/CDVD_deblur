import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from .arches import conv1x1, conv3x3, conv5x5, actFunc
from .common import DDBPN, RRDB, conv_na, norm
from data.utils import normalize_reverse
from train.utils import Mgrad


# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='relu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


# Middle network of residual dense blocks
class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer, activation))
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
    def __init__(self, in_channels, out_channels, growthRate, num_layer, activation='relu'):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, out_channels, stride=2)

    def forward(self, x, h=None):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)
        

        return out


# Global spatio-temporal attention module
class GSA(nn.Module):
    def __init__(self, para):
        super(GSA, self).__init__()
        self.n_feats = para.n_features
        self.center = para.past_frames
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
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
        self.fusion = conv1x1(self.related_f * (5 * self.n_feats), self.related_f * (5 * self.n_feats))

    def forward(self, hs):
        # hs: [(n=4,c=80,h=64,w=64), ..., (n,c,h,w)]
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
        out = self.fusion(torch.cat(cor_l, dim=1))

        return out


# RDB-based RNN cell
class RDBCell(nn.Module):
    def __init__(self, para):
        super(RDBCell, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_B0 = conv5x5(3, 48, stride=1)
        self.F_B1 = RDB_DS(in_channels=48, out_channels=48, growthRate=self.n_feats, num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=48, out_channels=64, growthRate=int(self.n_feats * 3 / 2), num_layer=3,
                           activation=self.activation)
        # self.F_R = DDBPN(num_channels=80, base_filter=16, feat = 16, num_stages=8, scale_factor=4) ###D-DBPN
        self.F_R = RDNet(in_channels=(1 + 4) * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks, activation=self.activation)  # in: 80
        # F_h: hidden state part
        self.F_h = nn.Sequential(
            conv3x3(80, self.n_feats),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation),
            conv3x3(self.n_feats, self.n_feats)
        )

    def forward(self, x, s_last):
        out = self.F_B0(x)
        out = self.F_B1(out)
        out = self.F_B2(out)
        out = torch.cat([out, s_last], dim=1)
        
        out = self.F_R(out)
        s = self.F_h(out)

        return out, s


# Reconstructor
class Reconstructor(nn.Module):
    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.para = para
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = para.n_features
        self.model = nn.Sequential(
            nn.ConvTranspose2d((5 * self.n_feats) * (self.related_f), 2 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv5x5(self.n_feats, 3*4, stride=1)
        )

    def forward(self, x):
        return self.model(x)


# Refiner
class Refiner(nn.Module):
    def __init__(self, para):
        super(Refiner, self).__init__()
        self.para = para
        self.eps = para.epsilon
        self.hd_project = conv1x1(15,15,stride=1)
        self.f_block1 = RRDB(15, 15, reduction=1, do_attention=True)
        self.f_concat = conv3x3(15,3)
        self.f_gbout = conv1x1(3,3)
        self.hd_project2 = conv1x1(6,12,stride=1)
        self.f_block2 = RRDB(12, 12, reduction=1, do_attention=True)
        self.f_dbout = conv_na(12, 12, kernel_size=3, padding=1, act_type='relu', norm_type=None)
        self.f_dbout2 = conv_na(12, 3, kernel_size=3, padding=1, act_type=None, norm_type=None)
    
    def forward(self, inputs, priors): #prior:(n,48,h,w)
        n,j,c,h,w = inputs.shape #n*4,3,h,w
        inputs = inputs.reshape(n*j,c,h,w)
        priors = priors.reshape(n,j,12,h,w).reshape(n*j,12,h,w)
        priors = torch.abs(priors) # p+n --> |p+n|
        G_inputs = Mgrad(inputs, self.eps) #n*4,3,h,w
        gbranch = self.hd_project(torch.cat([G_inputs, priors], dim=1))
        gbranch = self.f_block1(gbranch)
        gbranch = self.f_concat(gbranch) #n*4,3,h,w
        gbout = self.f_gbout(gbranch)
        dbranch = self.hd_project2(torch.cat([inputs, gbranch], dim=1))
        dbranch = self.f_block2(dbranch)
        dbout = self.f_dbout(dbranch)
        dbout = self.f_dbout2(dbout)
        return gbout.reshape(n,j,c,h,w).unsqueeze(1),  dbout.reshape(n,j,c,h,w).unsqueeze(1)

class Model(nn.Module):
    """
    Event-based ESTRNN
    """
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.ds_ratio = 4
        self.device = torch.device('cuda')
        self.cell = RDBCell(para)
        self.recons = Reconstructor(para)
        self.fusion = GSA(para)
        # for param in self.cell.parameters():
        #     param.requires_grad = False
        #     # print("cell param.requires_grad1: {}".format(param.requires_grad))
        # for param in self.recons.parameters():
        #     param.requires_grad = False
        #     # print("recons param.requires_grad1: {}".format(param.requires_grad))
        # for param in self.fusion.parameters():
        #     param.requires_grad = False
        #     # print("fusion param.requires_grad1: {}".format(param.requires_grad))
        # print("estrnn para,.requires_grad: False")
        self.refine = Refiner(para)

    def forward(self, x):
        outputs, hs = [], []
        batch_size, frames, channels, height, width = x.shape
        blurry = x[:,self.num_fb:(frames - self.num_ff),0:3,:,:]
        warpedsbt = x[:,self.num_fb:(frames - self.num_ff),7:,:,:]
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        # forward h structure: (batch_size, channel, height, width)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            h, s = self.cell(x[:, i, 0:3, :, :], s)
            hs.append(h)
        for i in range(self.num_fb, frames - self.num_ff):
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            outputs.append(out.unsqueeze(dim=1))
        outputs = torch.cat(outputs, dim=1)
        n,f,c,h,w = outputs.shape
        outputs = outputs.reshape(n,f,4,3,h,w) #n,f-m,4,3,h,w
        outputs += blurry.unsqueeze(2).repeat(1,1,4,1,1,1)
#         print('outputs shape', outputs.shape)
        outputs_refine=[]
        gbouts = []
        for i in range(f):
            gbout, output_refine = self.refine(outputs[:,i], warpedsbt[:,i])
            outputs_refine.append(output_refine)
            gbouts.append(gbout)
        outputs_refine = torch.cat(outputs_refine, dim=1)#n,f-m,4,3,h,w
        outputs_refine = normalize_reverse(outputs_refine, normalize=True)#n,f-m,4,3,h,w
        gbouts = torch.cat(gbouts, dim=1)#n,f-m,4,3,h,w

        outputs={}
        outputs['gbouts_norm'] = gbouts
        outputs['dbouts'] = outputs_refine

        return outputs


def feed(model, iter_samples):
    inputs = iter_samples[0]#n,f,3+4+4,h,w
    outputs = model(inputs)
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, seq_length, 55, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x,), verbose=False)

    return flops / seq_length, params
