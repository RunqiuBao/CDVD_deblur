import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from model.evflownet_basic_layers import *

_BASE_CHANNELS = 64


class Model(nn.Module):
    """
    EVFlow-Net
    """
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.device = torch.device('cuda')
        self.encoder1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self.para.no_batch_norm)
        self.encoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self.para.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self.para.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self.para.no_batch_norm)

        self.resnet_block = nn.Sequential(*[build_resnet_block(8*_BASE_CHANNELS, do_batch_norm=not self.para.no_batch_norm) for i in range(2)])

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                        out_channels=4*_BASE_CHANNELS, do_batch_norm=not self.para.no_batch_norm)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, do_batch_norm=not self.para.no_batch_norm)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=_BASE_CHANNELS, do_batch_norm=not self.para.no_batch_norm)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self.para.no_batch_norm) 
        self.mywarp = ForwardWarp_weight()

    def forward(self, inputs, profile_flag=False):
        
        sbt = inputs[:,0:2,:,:].clone()#(4,2,h,w)
        tsi = inputs[:,2:,:,:].clone()#(4,2,h,w)
        
       # encoder
        skip_connections = {}
        inputs = self.encoder1(inputs)
        skip_connections['skip0'] = inputs.clone()
        inputs = self.encoder2(inputs)
        skip_connections['skip1'] = inputs.clone()
        inputs = self.encoder3(inputs)
        skip_connections['skip2'] = inputs.clone()
        inputs = self.encoder4(inputs)
        skip_connections['skip3'] = inputs.clone()

        # transition
        inputs = self.resnet_block(inputs)

        # decoder
        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_dict['flow0'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_dict['flow1'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_dict['flow2'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict['flow3'] = flow.clone()
        
#         print("flow0 max min mean median: {}, {}, {}, {}".format(flow_dict['flow0'].max(), flow_dict['flow0'].min(), flow_dict['flow0'].mean(), flow_dict['flow0'].median()))
#         print("flow1 max min mean median: {}, {}, {}, {}".format(flow_dict['flow1'].max(), flow_dict['flow1'].min(), flow_dict['flow1'].mean(), flow_dict['flow1'].median()))
#         print("flow2 max min mean median: {}, {}, {}, {}".format(flow_dict['flow2'].max(), flow_dict['flow2'].min(), flow_dict['flow2'].mean(), flow_dict['flow2'].median()))
#         print("flow3 max min mean median: {}, {}, {}, {}".format(flow_dict['flow3'].max(), flow_dict['flow3'].min(), flow_dict['flow3'].mean(), flow_dict['flow3'].median()))
        
        # warping the sbt to t_start with the optical flow
        warped = {}
        for i in range(4):
            forward_optical_flow = flow_dict['flow{}'.format(i)].clone()
            tsi_resize = tsi.clone()
            sbt_resize = sbt.clone()
            n,c,h,w = tsi_resize.shape
            tsi_resize = F.interpolate(tsi_resize, (int(h/2**(3-i)), int(w/2**(3-i))), mode='nearest')
            sbt_resize = F.interpolate(sbt_resize, (int(h/2**(3-i)), int(w/2**(3-i))), mode='nearest')
            
            forward_move_p = forward_optical_flow * tsi_resize[:,0,:,:].unsqueeze(1)
             # restrict optical flow
            of_min = -20/2**(3-i)
            of_max = 20/2**(3-i)
            forward_move_p = torch.clamp(forward_move_p, min=of_min, max=of_max)
            warpresult_p, _ = self.mywarp.forward(tsi_resize[:,0,:,:].unsqueeze(1), sbt_resize[:,0,:,:].unsqueeze(1), forward_move_p) #(4,2,h,w), (4,2,h,w)
            forward_move_n = forward_optical_flow * tsi_resize[:,1,:,:].unsqueeze(1)
            forward_move_n = torch.clamp(forward_move_n, min=of_min, max=of_max)
            warpresult_n, _ = self.mywarp.forward(tsi_resize[:,1,:,:].unsqueeze(1), sbt_resize[:,1,:,:].unsqueeze(1), forward_move_n)
            warped['2start_{}'.format(i)] = torch.cat([warpresult_p, warpresult_n], dim=1)
            
            # flow reversal
            full_time_temp = torch.ones((n,1,int(h/2**(3-i)), int(w/2**(3-i)))).cuda()*13/12
            tsi_resize_p = tsi_resize[:,0,:,:].unsqueeze(1)
            full_time_temp[tsi_resize_p==0]=0
            reverse_move_p = forward_optical_flow * (tsi_resize_p - full_time_temp)
            reverse_move_p = torch.clamp(reverse_move_p, min=of_min, max=of_max)
            warpresult_p, _ = self.mywarp.forward(tsi_resize[:,0,:,:].unsqueeze(1), sbt_resize[:,0,:,:].unsqueeze(1), reverse_move_p)
            
            if i==3:
                flow_dict['reverse_pixelmove3'] = reverse_move_p
                flow_dict['pixelmove3'] = forward_move_p
                
            full_time_temp = torch.ones((n,1,int(h/2**(3-i)), int(w/2**(3-i)))).cuda()*13/12
            tsi_resize_n = tsi_resize[:,1,:,:].unsqueeze(1)
            full_time_temp[tsi_resize_n==0]=0
            reverse_move_n = forward_optical_flow * (tsi_resize_n-full_time_temp)
            reverse_move_n = torch.clamp(reverse_move_n, min=of_min, max=of_max)
            warpresult_n, _ = self.mywarp.forward(tsi_resize[:,1,:,:].unsqueeze(1), sbt_resize[:,1,:,:].unsqueeze(1), reverse_move_n)
            warped['2end_{}'.format(i)] = torch.cat([warpresult_p, warpresult_n], dim=1)
        return flow_dict, warped #4 pairs, each (4,2,h,w), (4,2,h,w)


def feed(model, iter_samples):
    inputs = iter_samples #4,2x2,h,w
    outputs = model(inputs.squeeze())
#     print("inside feed!")
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(4, 4, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x,profile_flag), verbose=False)

    return flops / seq_length, params
