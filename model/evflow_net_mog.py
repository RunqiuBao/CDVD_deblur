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
        self.mywarp = ForwardWarp()

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
#         print("flow1 max min mean median: {}, {}, {}, {}".format(flow_dict['flow1'].max(), flow_dict['flow1'].min(), flow_dict['flow0'].mean(), flow_dict['flow0'].median()))
#         print("flow2 max min mean median: {}, {}, {}, {}".format(flow_dict['flow2'].max(), flow_dict['flow2'].min(), flow_dict['flow0'].mean(), flow_dict['flow0'].median()))
#         print("flow3 max min mean median: {}, {}, {}, {}".format(flow_dict['flow3'].max(), flow_dict['flow3'].min(), flow_dict['flow0'].mean(), flow_dict['flow0'].median()))
        
        # warping the sbt to t_start with the optical flow
        warped = {}
        for i in range(4):
            forward_optical_flow = flow_dict['flow{}'.format(i)].clone()
            tsi_resize = tsi.clone()
            sbt_resize = sbt.clone()
            n,c,h,w = tsi_resize.shape
            tsi_resize = F.interpolate(tsi_resize, (int(h/2**(3-i)), int(w/2**(3-i))), mode='nearest')
            sbt_resize = F.interpolate(sbt_resize, (int(h/2**(3-i)), int(w/2**(3-i))), mode='nearest')
            
            # restrict optical flow
#             of_min = -10/2**(3-i)
#             of_max = 10/2**(3-i)
#             forward_move = torch.clamp(forward_move, min=of_min, max=of_max)
            warpresult_p, _ = self.mywarp.forward(sbt_resize[:,0,:,:].unsqueeze(1), forward_optical_flow * tsi_resize[:,0,:,:].unsqueeze(1)) #(4,1,h,w), (4,2,h,w)
            warpresult_n, _ = self.mywarp.forward(sbt_resize[:,1,:,:].unsqueeze(1), forward_optical_flow * tsi_resize[:,1,:,:].unsqueeze(1)) #(4,1,h,w), (4,2,h,w)
#             warpresult[norms > 0] = warpresult[norms > 0]/norms[norms>0].clone() #qvi的源代码里有这一步，但是导致warp的结果不够sharp，暂时不用。
            warped['2start_{}'.format(i)] = torch.cat([warpresult_p,warpresult_n], dim=1) 
                    
        
            # flow reversal
            forward_optical_flow_reverse = -forward_optical_flow # optical flow at t_start to t_end
            full_time_temp = torch.ones((n,1, int(h/2**(3-i)),int(w/2**(3-i)))).cuda()*13/12
            tsi_resize_p = tsi_resize[:,0,:,:].unsqueeze(1)
            full_time_temp[tsi_resize_p==0]=0
            tsi_resize_reverse = full_time_temp - tsi_resize_p
            reverse_move = forward_optical_flow_reverse * tsi_resize_reverse
            warpresult3_p, _ = self.mywarp.forward(sbt_resize[:,0,:,:].unsqueeze(1), reverse_move) #(4,1,h,w), (4,2,h,w)
            if i==3:
                if self.para.test_only == True:
#                     warped['2end_3_sbt'] = warpresult3_p
                    flow_dict['reverse_pixelmove3'] = reverse_move
#                     warped['2start_3_sbt'] = warpresult_p
                    flow_dict['pixelmove3'] = forward_optical_flow * tsi_resize[:,0,:,:].unsqueeze(1)
            full_time_temp = torch.ones((n,1,int(h/2**(3-i)),int(w/2**(3-i)))).cuda()*13/12
            tsi_resize_n = tsi_resize[:,1,:,:].unsqueeze(1)
            full_time_temp[tsi_resize_n==0]=0
            tsi_resize_reverse = full_time_temp - tsi_resize_n
            reverse_move = forward_optical_flow_reverse * tsi_resize_reverse
            warpresult3_n, _ = self.mywarp.forward(sbt_resize[:,1,:,:].unsqueeze(1), reverse_move)
            warped['2end_{}'.format(i)] = torch.cat([warpresult3_p, warpresult3_n], dim=1) #(4,2,h,w)
                        
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
'''
warp one-directional

warped = {}
        for i in range(4):
            forward_optical_flow = flow_dict['flow{}'.format(i)].clone()
            tsi_resize = tsi.clone()
            sbt_resize = sbt.clone()
            n,c,h,w = tsi_resize.shape
            tsi_resize = F.interpolate(tsi_resize, (int(h/2**(3-i)), int(w/2**(3-i))), mode='nearest')
            sbt_resize = F.interpolate(sbt_resize, (int(h/2**(3-i)), int(w/2**(3-i))), mode='nearest')
            
            
            # restrict optical flow
#             of_min = -10/2**(3-i)
#             of_max = 10/2**(3-i)
#             forward_move = torch.clamp(forward_move, min=of_min, max=of_max)
            warpresult_p, _ = self.mywarp.forward(sbt_resize[:,0,:,:].unsqueeze(1), forward_optical_flow * tsi_resize[:,0,:,:].unsqueeze(1)) #(4,1,h,w), (4,2,h,w)
            warpresult_n, _ = self.mywarp.forward(sbt_resize[:,1,:,:].unsqueeze(1), forward_optical_flow * tsi_resize[:,1,:,:].unsqueeze(1)) #(4,1,h,w), (4,2,h,w)
#             warpresult[norms > 0] = warpresult[norms > 0]/norms[norms>0].clone() #qvi的源代码里有这一步，但是导致warp的结果不够sharp，暂时不用。
            warped['2start_{}'.format(i)] = torch.cat([warpresult_p,warpresult_n], dim=1) 
                    
            if i==3:
                if self.para.test_only == True:
                    flow_dict['pixelmove3'] = forward_optical_flow * tsi_resize[:,0,:,:].unsqueeze(1)
                        
        return flow_dict, warped #4 pairs, each (4,2,h,w), (4,2,h,w)


'''


'''
warp non-polarity sbt:

# warping the sbt to t_start with the optical flow
        warped = {}
        for i in range(4):
            forward_optical_flow = flow_dict['flow{}'.format(i)].clone()
            tsi_resize = tsi.clone()
            sbt_resize = sbt.clone()
            n,c,h,w = tsi_resize.shape
            tsi_resize = F.interpolate(tsi_resize, (int(h/2**(3-i)), int(w/2**(3-i))), mode='nearest')
            sbt_resize = F.interpolate(sbt_resize, (int(h/2**(3-i)), int(w/2**(3-i))), mode='nearest')
            sbt_resize = sbt_resize[:,0,:,:].unsqueeze(1) #torch.sum(sbt_resize, dim=1).unsqueeze(1) #(4,1,h,w)
            tsi_resize = tsi_resize[:,0,:,:].unsqueeze(1) #torch.where(sbt_resize>0, tsi_resize[:,0,:,:].unsqueeze(1), tsi_resize[:,1,:,:].unsqueeze(1))
            
            forward_move = forward_optical_flow.clone() * tsi_resize.clone()
            # restrict optical flow
#             of_min = -5/2**(3-i)
#             of_max = 5/2**(3-i)
#             forward_move = torch.clamp(forward_move, min=of_min, max=of_max)
            warpresult, norms = self.mywarp.forward(sbt_resize.clone(), forward_move) #(4,1,h,w), (4,2,h,w)
#             warpresult[norms > 0] = warpresult[norms > 0]/norms[norms>0].clone() #qvi的源代码里有这一步，但是导致warp的结果不够sharp，暂时不用。
            warped['2start_{}'.format(i)] = warpresult
            if i==3:
                if self.para.test_only == True:
                    flow_dict['pixelmove3'] = forward_move
        
            # flow reversal
            forward_optical_flow_reverse = -forward_optical_flow.clone() # optical flow at t_start to t_end
            full_time_temp = torch.ones((n,1, int(h/2**(3-i)),int(w/2**(3-i)))).cuda()*13/12
            full_time_temp[tsi_resize_reverse==0]=0
            tsi_resize_reverse = full_time_temp - tsi_resize.clone()
            reverse_move = forward_optical_flow_reverse * tsi_resize_reverse
            warpresult3, norms3 = self.mywarp.forward(sbt_resize.clone(), reverse_move) #(4,1,h,w), (4,2,h,w)
#             warpresult3[norms3 > 0] = warpresult3[norms3 > 0]/norms3[norms3>0].clone() 
            warped['2end_{}'.format(i)] = warpresult3 #(4,1,h,w)
            if i==3:
                flow_dict['reverse_pixelmove3'] = reverse_move
'''