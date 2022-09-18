import os
from os.path import join, dirname

import cv2
import numpy as np
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# todo in case of dict
def reduce_tensor(num_gpus, ts):
    """
    reduce tensor from multiple gpus
    """
    # todo loss of ddp mode
    if isinstance(ts, dict):
        raise NotImplementedError
    else:
        try:
            dist.reduce(ts, dst=0, op=dist.ReduceOp.SUM)
            ts /= num_gpus
        except:
            msg = '{}'.format(type(ts))
            raise NotImplementedError(msg)
    return ts


def img2video(path, size, seq, frame_start, frame_end, marks, fps=10):
    """
    generate video
    """
    file_path = join(path, '{}.avi'.format(seq))
    os.makedirs(dirname(path), exist_ok=True)
    path = join(path, '{}'.format(seq))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video = cv2.VideoWriter(file_path, fourcc, fps, size)
    for i in range(frame_start, frame_end):
        imgs = []
        for j in range(len(marks)):
            img_path = join(path, '{:08d}_{}.png'.format(i, marks[j].lower()))
            img = cv2.imread(img_path)
            img = cv2.putText(img, marks[j], (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            imgs.append(img)
        frame = np.concatenate(imgs, axis=1)
        video.write(frame)
    video.release()

def Mgrad(inputs, epsilon=1e-6, ifmog=True): #(n,3,h,w)
    n,c,h,w = inputs.shape
    # soble kernel according to kornia
    sobel_x = torch.Tensor([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
    ]).view((1,1,3,3)).float()
    weight_x = nn.Parameter(data=sobel_x, requires_grad=False).cuda()
    sobel_y = torch.Tensor([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ]).view((1,1,3,3)).float()
    weight_y = nn.Parameter(data=sobel_y, requires_grad=False).cuda()
    Gs=[]
    for i in range(n):
        Gx_0 = F.conv2d(inputs[i,0,:,:].unsqueeze(0).unsqueeze(0), weight_x, padding=1)
        Gx_1 = F.conv2d(inputs[i,1,:,:].unsqueeze(0).unsqueeze(0), weight_x, padding=1)
        Gx_2 = F.conv2d(inputs[i,2,:,:].unsqueeze(0).unsqueeze(0), weight_x, padding=1)
        Gy_0 = F.conv2d(inputs[i,0,:,:].unsqueeze(0).unsqueeze(0), weight_y, padding=1)
        Gy_1 = F.conv2d(inputs[i,1,:,:].unsqueeze(0).unsqueeze(0), weight_y, padding=1)
        Gy_2 = F.conv2d(inputs[i,2,:,:].unsqueeze(0).unsqueeze(0), weight_y, padding=1)
        if ifmog:
            G0 = torch.sqrt(torch.pow(Gx_0,2)+torch.pow(Gy_0,2)+epsilon)
            G1 = torch.sqrt(torch.pow(Gx_1,2)+torch.pow(Gy_1,2)+epsilon)
            G2 = torch.sqrt(torch.pow(Gx_2,2)+torch.pow(Gy_2,2)+epsilon)
        else:
            G0 = torch.cat([Gx_0,Gy_0],dim=1)
            G1 = torch.cat([Gx_1,Gy_1],dim=1)
            G2 = torch.cat([Gx_2,Gy_2],dim=1)
        G = torch.cat([G0,G1,G2], dim=1)
        Gs.append(G)
    Gs = torch.cat(Gs, dim=0)
    return Gs

def Mgrad_mono(inputs, epsilon=1e-6, ifmog=True): #(n,1,h,w)
    n,c,h,w = inputs.shape
    # soble kernel according to kornia
    sobel_x = torch.Tensor([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
    ]).view((1,1,3,3)).float()
    weight_x = nn.Parameter(data=sobel_x, requires_grad=False).cuda()
    sobel_y = torch.Tensor([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ]).view((1,1,3,3)).float()
    weight_y = nn.Parameter(data=sobel_y, requires_grad=False).cuda()
    Gs=[]
    for i in range(n):
        Gx_0 = F.conv2d(inputs[i,0,:,:].unsqueeze(0).unsqueeze(0), weight_x, padding=1)
        Gy_0 = F.conv2d(inputs[i,0,:,:].unsqueeze(0).unsqueeze(0), weight_y, padding=1)
        if ifmog:
            G0 = torch.sqrt(torch.pow(Gx_0,2)+torch.pow(Gy_0,2)+epsilon)
        else:
            G0 = torch.cat([Gx_0,Gy_0],dim=1)
        G = G0
        Gs.append(G)
    Gs = torch.cat(Gs, dim=0)
    return Gs



