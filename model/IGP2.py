import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from .arches import conv1x1, conv3x3, conv5x5, actFunc
from data.utils import normalize_reverse
from .igp_compo_2 import IGP2

class Model(nn.Module):
    """
    IGP
    """
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.igp2 = IGP2(para)
        self.device = torch.device('cuda')

    def forward(self, x):
        n,fm,c,h,w = x.shape
        x = x[:,:,1:,:,:]
        outputs = self.igp2(x)
        return outputs


def feed(model, iter_samples):
    inputs = iter_samples[0]#n,f,3+4+4,h,w
    outputs = model(inputs)
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, seq_length,17, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x,), verbose=False)

    return flops / seq_length, params
