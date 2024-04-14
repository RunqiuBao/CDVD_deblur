from importlib import import_module

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        model_name = para.model
        self.module = import_module('model.{}'.format(model_name))
        self.model = self.module.Model(para)

    def forward(self, iter_samples):
        sbt_out = self.module.feed(self.model, iter_samples)
        return sbt_out

    def profile(self):
        H, W = self.para.profile_H, self.para.profile_W
#         seq_length = self.para.future_frames + self.para.past_frames + 1
        seq_length = 5
        flops, params = self.module.cost_profile(self.model, H, W, seq_length, self.para)
        return flops, params
