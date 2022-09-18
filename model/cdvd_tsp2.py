import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from model import cdvd_recons_video
from model import flow_pwc
from utils import cdvd_utils
from data.utils import normalize_reverse


class Model(nn.Module):
    """
    CDVD_TSP
    """
    def __init__(self, para, in_channels=3, n_sequence=5, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=True, load_recons_net=False, flow_pretrain_fn='/home/runqiu/code/CDVD_ESTRNN/pretrain_model/network-default.pytorch', recons_pretrain_fn='',
                 is_mask_filter=True, device='cuda'):
        super(Model, self).__init__()
        print("Creating CDVD-TSP Net")

        self.n_sequence = n_sequence
        self.device = device

        assert n_sequence == 5, "Only support args.n_sequence=5; but get args.n_sequence={}".format(n_sequence)

        self.is_mask_filter = is_mask_filter
        print('Is meanfilter image when process mask:', 'True' if is_mask_filter else 'False')
        extra_channels = 1
        print('Select mask mode: concat, num_mask={}'.format(extra_channels))

        self.flow_net = flow_pwc.Flow_PWC(load_pretrain=load_flow_net, pretrain_fn=flow_pretrain_fn, device=device)
        self.recons_net = cdvd_recons_video.RECONS_VIDEO(in_channels=in_channels, n_sequence=3, out_channels=out_channels,
                                                    n_resblock=n_resblock, n_feat=n_feat,
                                                    extra_channels=extra_channels)
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))

    def get_masks(self, img_list, flow_mask_list):
        num_frames = len(img_list)

        img_list_copy = [img.detach() for img in img_list]  # detach backward
        if self.is_mask_filter:  # mean filter
            img_list_copy = [cdvd_utils.calc_meanFilter(im, n_channel=3, kernel_size=5) for im in img_list_copy]

        delta = 1.
        mid_frame = img_list_copy[num_frames // 2]
        diff = torch.zeros_like(mid_frame)
        for i in range(num_frames):
            diff = diff + (img_list_copy[i] - mid_frame).pow(2)
        diff = diff / (2 * delta * delta)
        diff = torch.sqrt(torch.sum(diff, dim=1, keepdim=True))
        luckiness = torch.exp(-diff)  # (0,1)

        sum_mask = torch.ones_like(flow_mask_list[0])
        for i in range(num_frames):
            sum_mask = sum_mask * flow_mask_list[i]
        sum_mask = torch.sum(sum_mask, dim=1, keepdim=True)
        sum_mask = (sum_mask > 0).float()
        luckiness = luckiness * sum_mask

        return luckiness

    def forward(self, x):
        # (n,5,17,h,w)
        x = x[:,:,0,:,:].unsqueeze(2)
        x = x.repeat(1,1,3,1,1)

        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]

        # Interation 1
        warped01, _, _, flow_mask01 = self.flow_net(frame_list[1], frame_list[0])
        warped21, _, _, flow_mask21 = self.flow_net(frame_list[1], frame_list[2])
        warped12, _, _, flow_mask12 = self.flow_net(frame_list[2], frame_list[1])
        warped32, _, _, flow_mask32 = self.flow_net(frame_list[2], frame_list[3])
        warped23, _, _, flow_mask23 = self.flow_net(frame_list[3], frame_list[2])
        warped43, _, _, flow_mask43 = self.flow_net(frame_list[3], frame_list[4])
        one_mask = torch.ones_like(flow_mask01)

        frame_warp_list = [warped01, frame_list[1], warped21]
        flow_mask_list = [flow_mask01, one_mask.detach(), flow_mask21]
        luckiness = self.get_masks(frame_warp_list, flow_mask_list)
        concated = torch.cat([warped01, frame_list[1], warped21, luckiness], dim=1)
        recons_1, _ = self.recons_net(concated)

        frame_warp_list = [warped12, frame_list[2], warped32]
        flow_mask_list = [flow_mask12, one_mask.detach(), flow_mask32]
        luckiness = self.get_masks(frame_warp_list, flow_mask_list)
        concated = torch.cat([warped12, frame_list[2], warped32, luckiness], dim=1)
        recons_2, _ = self.recons_net(concated)

        frame_warp_list = [warped23, frame_list[3], warped43]
        flow_mask_list = [flow_mask23, one_mask.detach(), flow_mask43]
        luckiness = self.get_masks(frame_warp_list, flow_mask_list)
        concated = torch.cat([warped23, frame_list[3], warped43, luckiness], dim=1)
        recons_3, _ = self.recons_net(concated)

        # Interation 2
        warped_recons12, _, _, flow_mask_recons12 = self.flow_net(recons_2, recons_1)
        warped_recons32, _, _, flow_mask_recons32 = self.flow_net(recons_2, recons_3)
        frame_warp_list = [warped_recons12, recons_2, warped_recons32]
        flow_mask_list = [flow_mask_recons12, one_mask.detach(), flow_mask_recons32]
        luckiness = self.get_masks(frame_warp_list, flow_mask_list)
        concated = torch.cat([warped_recons12, recons_2, warped_recons32, luckiness], dim=1)
        out, _ = self.recons_net(concated)

        recons_1 = normalize_reverse(recons_1, normalize=True)
        recons_2 = normalize_reverse(recons_2, normalize=True)
        recons_3 = normalize_reverse(recons_3, normalize=True)
        out = normalize_reverse(out, normalize=True)

        return recons_1[:,0].unsqueeze(1), recons_2[:,0].unsqueeze(1), recons_3[:,0].unsqueeze(1), out[:,0].unsqueeze(1)   # (n,1,h,w)


def feed(model, iter_samples):
    inputs = iter_samples[0]  # (n,f,3+4,h,w)
    outputs = model(inputs)
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, seq_length, 17, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x,), verbose=False)

    return flops / seq_length, params
