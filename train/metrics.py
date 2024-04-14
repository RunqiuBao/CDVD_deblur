# import lpips
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as compare_ssim
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from data.utils import normalize_reverse
from train.utils import Mgrad
from pytorch_msssim import ssim, ms_ssim

def estimate_mask(img):
    mask = img.copy()
    mask[mask > 0.0] = 1.0
    return mask


def mask_pair(x, y, mask):
    return x * mask, y * mask


def im2tensor(image, cent=1., factor=255. / 2.):
    image = image.astype(np.float)
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


# def psnr_calculate(x, y, val_range=255.0):
#     # x,y size (h,w,c)
#     # assert len(x.shape) == 3
#     # assert len(y.shape) == 3
#     x = x.astype(np.float)
#     y = y.astype(np.float)
#     diff = (x - y) / val_range
#     mse = np.mean(diff ** 2)
#     psnr = -10 * np.log10(mse)
#     return psnr

def psnr_calculate(x, y, val_range=255.0):
    '''
    x, y can only be single image.
    '''
    # x,y size (h,w,c)
    # assert len(x.shape) == 3
    # assert len(y.shape) == 3
    x = x.astype(np.float)
    y = y.astype(np.float)
    diff = (x - y) / val_range
    mse = np.mean(diff ** 2)
    psnr = -10 * np.log10(mse)
    return psnr

def ssim_calc_torch(x,y,val_range=255.0):
    myssim = ssim( x, y, data_range=val_range, size_average=True) # return a scalar
    return myssim

def ssim_calculate(x, y, val_range=255.0):
    myssim = compare_ssim(y, x, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                        data_range=val_range)
    return myssim

def mog_calculate(flow_dict, warped):
    loss_all = None
    for i in range(4):
        X = flow_dict['flow{}'.format(i)]
        Y = warped['2start_{}'.format(i)]
        n,c,h,w = Y.shape
        sobel_x = torch.Tensor([
            [1,0,-1],
            [2,0,-2],
            [1,0,-1]
        ]).float().cuda()
        sobel_y = torch.Tensor([
             [1,2,1],
             [0,0,0],
             [-1,-2,-1]
        ]).float().cuda()
        if c==1:
            sobel_x = sobel_x.view((1,1,3,3))#.repeat(1,2,1,1)
            G_x = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_x, padding=1)

            sobel_y = sobel_y.view((1,1,3,3,))#.repeat(1,2,1,1)
            G_y = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_y, padding=1)

            G = torch.pow(G_x,2)+torch.pow(G_y,2)
            mask = torch.where(G>0)
            loss_mog = torch.mean(G[mask])
        elif c==2:
            sobel_x = sobel_x.view((1,1,3,3))
            G_x_p = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_x, padding=1)
            G_x_n = F.conv2d(Y[:,1,:,:].unsqueeze(1), sobel_x, padding=1)

            sobel_y = sobel_y.view((1,1,3,3,))
            G_y_p = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_y, padding=1)
            G_y_n = F.conv2d(Y[:,1,:,:].unsqueeze(1), sobel_y, padding=1)

            G_x = torch.cat([G_x_p, G_x_n], dim=1)
            G_y = torch.cat([G_y_p, G_y_n], dim=1)

            G = torch.pow(G_x,2)+torch.pow(G_y,2)
            mask = torch.where(G>0)
            loss_mog = torch.mean(G[mask])
        else:
            raise NotImplementedError         

        if loss_all == None:
            loss_all = loss_mog
        else:
            loss_all += loss_mog

    return  loss_all

def average_timestamp_calculate(flow_dict, warped):
    loss_all = None
    for i in range(4):
        X = flow_dict['flow{}'.format(i)]
        Y = warped['2start_{}'.format(i)]
        Y_reverse = warped['2end_{}'.format(i)]
        loss_tsi = Y[:,0,:,:]**2 + Y[:,1,:,:]**2 + Y_reverse[:,0,:,:]**2 + Y_reverse[:,1,:,:]**2 #4,1,h,w
        loss_tsi = torch.mean(loss_tsi)

        if loss_all == None:
            loss_all = loss_tsi
        else:
            loss_all += loss_tsi
            
    return loss_all


# def lpips_calculate(x, y, net='alex', gpu=False):
#     # input range is 0~255
#     # image should be RGB, and normalized to [-1,1]
#     x = im2tensor(x[:, :, ::-1])
#     y = im2tensor(y[:, :, ::-1])
#     loss_fn = lpips.LPIPS(net=net, verbose=False)
#     if gpu:
#         x = x.cuda()
#         y = y.cuda()
#         loss_fn = loss_fn.cuda()
#     lpips_value = loss_fn(x, y)
#     return lpips_value.item()


class PSNR_shave(_Loss):
    def __init__(self, centralize=True, normalize=True, val_range=255.):
        super(PSNR_shave, self).__init__()
        self.centralize = centralize
        self.normalize = normalize
        self.val_range = val_range

    def _quantize(self, img):
        img = img.clamp(0, self.val_range).round()
        return img

    def forward(self, x, y):
        
        if x.dim() == 3:
            n = 1
        elif x.dim() == 4:
            n = x.size(0)
        elif x.dim() == 5:
            n = x.size(0) * x.size(1)
            nold,fm,c,h,w = x.shape
            x = x.view(n,c,h,w)
            y=y.view(n,c,h,w)
        elif x.dim() == 6:
            n = x.size(0) * x.size(1) * x.size(2)
            nold,fm,c1,c2,h,w = x.shape
            # print("x shape: {}, yshape: {}, n: {}".format(x.shape, y.shape, n))
            x = x.view(n,c2,h,w)
            y=y.reshape(n,c2,h,w)
        else:
            raise NotImplementedError
        mses = []

        shave=4
        x = x[:,:,shave:-shave,shave:-shave]
        y = y[:,:,shave:-shave,shave:-shave]
        
        for i in range(n):
            diff = self._quantize(x[i]) - self._quantize(y[i])
            mse = diff.div(self.val_range).pow(2).view(1, -1).mean(dim=-1)
            mses.append(mse)
        mses = torch.cat(mses, dim=0)
        psnr = -10 * mse.log10()

        return psnr.mean()
    
class MoG(_Loss):
    def __init__(self):
        super(MoG, self).__init__()

    def forward(self, flow_dict, warped):
        loss_all = None
        for i in range(4):
            X = flow_dict['flow{}'.format(i)]
            Y = warped['2start_{}'.format(i)]
            Y_reverse = warped['2end_{}'.format(i)]
            n,c,h,w = Y.shape
            sobel_x = torch.Tensor([
                [1,0,-1],
                [2,0,-2],
                [1,0,-1]
            ]).float().cuda()
            sobel_y = torch.Tensor([
                 [1,2,1],
                 [0,0,0],
                 [-1,-2,-1]
            ]).float().cuda()
            if c==1:
                sobel_x = sobel_x.view((1,1,3,3))#.repeat(1,2,1,1)
                G_x = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_x, padding=1)
                G_x_reverse = F.conv2d(Y_reverse[:,0,:,:].unsqueeze(1), sobel_x, padding=1)

                sobel_y = sobel_y.view((1,1,3,3,))#.repeat(1,2,1,1)
                G_y = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_y, padding=1)
                G_y_reverse = F.conv2d(Y_reverse[:,0,:,:].unsqueeze(1), sobel_y, padding=1)

                G = torch.pow(G_x,2)+torch.pow(G_y,2)
                mask = torch.where(G>0)
                loss_mog = torch.mean(G[mask])

                G_reverse = torch.pow(G_x_reverse,2)+torch.pow(G_y_reverse,2)
                mask = torch.where(G_reverse>0)
                loss_mog+=torch.mean(G_reverse[mask])
            elif c==2:
                sobel_x = sobel_x.view((1,1,3,3))
                G_x_p = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_x, padding=1)
                G_x_n = F.conv2d(Y[:,1,:,:].unsqueeze(1), sobel_x, padding=1)
                G_x_p_reverse = F.conv2d(Y_reverse[:,0,:,:].unsqueeze(1), sobel_x, padding=1)
                G_x_n_reverse = F.conv2d(Y_reverse[:,1,:,:].unsqueeze(1), sobel_x, padding=1)
                
                sobel_y = sobel_y.view((1,1,3,3,))
                G_y_p = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_y, padding=1)
                G_y_n = F.conv2d(Y[:,1,:,:].unsqueeze(1), sobel_y, padding=1)
                G_y_p_reverse = F.conv2d(Y_reverse[:,0,:,:].unsqueeze(1), sobel_y, padding=1)
                G_y_n_reverse = F.conv2d(Y_reverse[:,1,:,:].unsqueeze(1), sobel_y, padding=1)
                
                G_x = torch.cat([G_x_p, G_x_n], dim=1)
                G_x_reverse = torch.cat([G_x_p_reverse, G_x_n_reverse], dim=1)
                G_y = torch.cat([G_y_p, G_y_n], dim=1)
                G_y_reverse = torch.cat([G_y_p_reverse, G_y_n_reverse], dim=1)
                
                G = torch.pow(G_x,2)+torch.pow(G_y,2)
                mask = torch.where(G>0)
                loss_mog = torch.mean(G[mask])

                G_reverse = torch.pow(G_x_reverse,2)+torch.pow(G_y_reverse,2)
                mask = torch.where(G_reverse>0)
                loss_mog+=torch.mean(G_reverse[mask])
            else:
                raise NotImplementedError         
            
            if loss_all == None:
                loss_all = loss_mog
            else:
                loss_all += loss_mog
        
        return  loss_all
    
class MoG2(_Loss):
    def __init__(self):
        super(MoG2, self).__init__()

    def forward(self, flow_dict, warped):
        loss_all = None
        for i in range(4):
            X = flow_dict['flow{}'.format(i)]
            Y = warped['2start_{}'.format(i)]
            n,c,h,w = Y.shape
            sobel_x = torch.Tensor([
                [1,0,-1],
                [2,0,-2],
                [1,0,-1]
            ]).float().cuda()
            sobel_y = torch.Tensor([
                 [1,2,1],
                 [0,0,0],
                 [-1,-2,-1]
            ]).float().cuda()
            if c==1:
                sobel_x = sobel_x.view((1,1,3,3))#.repeat(1,2,1,1)
                G_x = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_x, padding=1)

                sobel_y = sobel_y.view((1,1,3,3,))#.repeat(1,2,1,1)
                G_y = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_y, padding=1)

                G = torch.pow(G_x,2)+torch.pow(G_y,2)
                mask = torch.where(G>0)
                loss_mog = torch.mean(G[mask])
            elif c==2:
                sobel_x = sobel_x.view((1,1,3,3))
                G_x_p = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_x, padding=1)
                G_x_n = F.conv2d(Y[:,1,:,:].unsqueeze(1), sobel_x, padding=1)
                
                sobel_y = sobel_y.view((1,1,3,3,))
                G_y_p = F.conv2d(Y[:,0,:,:].unsqueeze(1), sobel_y, padding=1)
                G_y_n = F.conv2d(Y[:,1,:,:].unsqueeze(1), sobel_y, padding=1)
                
                G_x = torch.cat([G_x_p, G_x_n], dim=1)
                G_y = torch.cat([G_y_p, G_y_n], dim=1)
                
                G = torch.pow(G_x,2)+torch.pow(G_y,2)
                mask = torch.where(G>0)
                loss_mog = torch.mean(G[mask])
            else:
                raise NotImplementedError         
            
            if loss_all == None:
                loss_all = loss_mog
            else:
                loss_all += loss_mog
        
        return  loss_all
    
class AverageTimestamp(_Loss):
    def __init__(self):
        super(AverageTimestamp, self).__init__()

    def forward(self, flow_dict, warped):
        loss_all = None
        for i in range(4):
            X = flow_dict['flow{}'.format(i)]
            Y = warped['2start_{}'.format(i)]
            Y_reverse = warped['2end_{}'.format(i)]
            loss_tsi = Y[:,0,:,:]**2 + Y[:,1,:,:]**2 + Y_reverse[:,0,:,:]**2 + Y_reverse[:,1,:,:]**2 #4,1,h,w
            loss_tsi = torch.mean(loss_tsi)
            
            if loss_all == None:
                loss_all = loss_tsi
            else:
                loss_all += loss_tsi
        
        return loss_all

class L1Distance(_Loss):
    def __init__(self, para) -> None:
        super(L1Distance, self).__init__()
        self.eps = para.epsilon
        self.l1 = nn.SmoothL1Loss()

    def forward(self, outputs, labels):
        '''
        outputs: (n,f,24,h,w)
        labels: (n*f*4,3,h,w)
        '''
        n,fm,_,h,w = outputs.shape
        j = 4
        c = 6
        # labels = labels.view(n*fm*j, c,h,w)
        outputs = outputs.reshape(n,fm,j,c,h,w).reshape(n*fm*j,c,h,w)
        for ii in range(n*fm*j):
            if ii==0:
                L_gm_l1 = self.l1(outputs[ii], labels[ii])
            else:
                L_gm_l1 += self.l1(outputs[ii], labels[ii])
        loss_all =L_gm_l1
        return loss_all

class L1Distance_mono(_Loss):
    def __init__(self, para) -> None:
        super(L1Distance_mono, self).__init__()
        self.eps = para.epsilon
        self.l1 = nn.SmoothL1Loss()

    def forward(self, outputs, labels):
        '''
        outputs: (n,f,8,h,w)
        labels: (n*f*4,2,h,w)
        '''
        n,fm,_,h,w = outputs.shape
        j = 4
        c = 2
        # labels = labels.view(n*fm*j, c,h,w)
        outputs = outputs.reshape(n,fm,j,c,h,w).reshape(n*fm*j,c,h,w)
        for ii in range(n*fm*j):
            if ii==0:
                L_gm_l1 = self.l1(outputs[ii], labels[ii])
            else:
                L_gm_l1 += self.l1(outputs[ii], labels[ii])
        loss_all =L_gm_l1
        return loss_all
