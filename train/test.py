import os
import pickle
import time
from os.path import join, dirname

from scipy.ndimage.measurements import label

import cv2
import lmdb
import numpy as np
import torch
import torch.nn as nn

from data.utils import normalize, normalize_reverse
from model import Model
from .metrics import psnr_calculate, ssim_calculate, mog_calculate, average_timestamp_calculate
from .utils import AverageMeter, img2video

from.utils import flow_to_image, flow_uv_to_colors, make_colorwheel
import copy
from train.utils import Mgrad, Mgrad_mono
from importlib import import_module

import matplotlib.pyplot as plt

def test(para, logger):
    """
    test code
    """
    # load model with checkpoint
    if not para.test_only:
        para.test_checkpoint = join(logger.save_dir, 'model_best.pth.tar')
    if para.test_save_dir is None:
        para.test_save_dir = logger.save_dir
    model = Model(para).cuda()
    checkpoint_path = para.test_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
    model = nn.DataParallel(model)
    # checkpoint['state_dict'] = {k:v for k,v in checkpoint['state_dict'].items() if 'igp2' in k}
    model.load_state_dict(checkpoint['state_dict'])

    ds_name = para.dataset
    logger('{} results generating ...'.format(ds_name), prefix='\n')
    ds_type = 'valid'
    # _test_lmdb(para, logger, model, ds_type) #event_estrnn
    # _test_lmdb4(para, logger, model, ds_type, ds_name) #igp
    # _test_lmdb(para, logger, model, ds_type, ds_name) #igp_ori
    _test_lmdb_cdvd(para, logger, model, ds_type, ds_name) #igp_ori

def _test_lmdb4(para, logger, model, ds_type, ds_name):
    L1Distance = AverageMeter()
    metrics_name = para.metrics
    module = import_module('train.metrics')
    # metrics = getattr(module, metrics_name)(para).cuda()
    # SSIM = AverageMeter()
    timer = AverageMeter()
    results_register = set()
    B, H, W, C = 1, 180, 240, 1
    
    path = join(para.data_root, ds_name)
    data_test_path = join(path, '{}_{}_blurry'.format(ds_name, ds_type))
    data_test_gt_path = join(path, '{}_{}_gt'.format(ds_name, ds_type))
    data_test_evt_path = join(path, '{}_{}_sbt'.format(ds_name, ds_type))
    env_blur = lmdb.open(data_test_path, map_size=1099511627776)
    env_gt = lmdb.open(data_test_gt_path, map_size=1099511627776)
    env_evt = lmdb.open(data_test_evt_path, map_size=1099511627776)
    txn_blur = env_blur.begin()
    txn_gt = env_gt.begin()
    txn_evt = env_evt.begin()

    data_test_info_path = join(path, '{}_info_{}.pkl'.format(ds_name, ds_type))
    with open(data_test_info_path, 'rb') as f:
        seqs_info = pickle.load(f)

    for seq_idx in range(seqs_info['num']):#
        # if seq_idx!=0 and seq_idx!=1 and seq_idx!=5:
        #     continue
        seq_length = seqs_info[seq_idx]['length']
        seq = '{:03d}'.format(seq_idx)
        logger('seq {} image results generating ...'.format(seq))
        dir_name = '_'.join((para.dataset, para.model, 'test'))
        save_dir = join(para.test_save_dir, dir_name, seq)
        os.makedirs(save_dir, exist_ok=True)
        start = 0
        end = para.test_frames

        input_seq = []
        label_seq = []
        event_seq = []
        label_seq_forinput = []
        for frame_idx in range(start, end):
            code = '%03d_%08d' % (seq_idx, frame_idx)
            code = code.encode()
            
            blur_img = txn_blur.get(code)
            blur_img = np.frombuffer(blur_img, dtype='uint8')
            blur_img = blur_img.reshape(H, W)[np.newaxis,np.newaxis,:,:].astype(np.float32)
            gt_img = txn_gt.get(code)
            gt_img = np.frombuffer(gt_img, dtype='uint8')
            gt_img = gt_img.reshape(4, H, W)[np.newaxis,:,:,:]
            event_stack = txn_evt.get(code)
            event_stack = np.frombuffer(event_stack, dtype='int8')
            event_stack = event_stack.reshape(16, H, W)[np.newaxis,:].astype(np.float32)
        
            event_seq.append(event_stack)
            input_seq.append(blur_img)
            label_seq.append(gt_img)
        input_seq = np.ascontiguousarray(np.concatenate(input_seq)) #10,c,h,w
        event_seq = np.ascontiguousarray(np.concatenate(event_seq))#10,4,h,w
        label_seq = np.ascontiguousarray(np.concatenate(label_seq))
        model.eval()
        with torch.no_grad():
            input_seq = torch.from_numpy(input_seq).cuda().unsqueeze(0)/255
            event_seq = torch.from_numpy(event_seq).cuda().unsqueeze(0)/255
        
            # print('input_seq',input_seq.max(),input_seq.min(),input_seq.mean())
            # print('event_seq',event_seq.max(),event_seq.min(),event_seq.mean())
            time_start = time.time()
            output_seq = model([torch.cat([input_seq, event_seq],dim=2), ])#[(1,10,48,h,w)]
            if isinstance(output_seq, (list, tuple)):
                output_seq = output_seq[0]
            timer.update((time.time() - time_start) / len(output_seq), n=len(output_seq))
        output_seq = output_seq.squeeze()
        fm,c,h,w = output_seq.shape
        output_seq = output_seq.reshape(fm, 4,2,h,w)
        label_seq = label_seq[para.past_frames:(para.test_frames - para.future_frames)]
        for frame_idx in range(para.test_frames-para.past_frames-para.future_frames):
            for j in range(4):
                gt_img = label_seq[frame_idx,j]
                gt_img = torch.from_numpy(gt_img.astype(np.float32)/255).unsqueeze(0).cuda()
                with torch.no_grad():
                    G = Mgrad_mono(gt_img[np.newaxis,:,:], para.epsilon, ifmog=False)#(1,2,h,w)
                    G_m = torch.sum(G[0], 0)
                    gt_img = G_m.squeeze().cpu().numpy()
                gt_img_path = join(save_dir, '{:08d}_gt_{}.png'.format(frame_idx + start, j))
#                 print('output_seq shape:', output_seq.shape)
                g_m = output_seq[frame_idx,j].detach()
                G = torch.sum(g_m, 0)
                gm_img = G.squeeze().cpu().numpy()
                gm_img_path = join(save_dir, '{:08d}_{}_{}.png'.format(frame_idx + start, j, para.model.lower()))
                
                fig = plt.figure(frameon=False)
                fig.set_size_inches(64,36)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(gt_img, aspect='auto')
                plt.savefig(gt_img_path, dpi=20)

                fig = plt.figure(frameon=False)
                fig.set_size_inches(64,36)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(gm_img, aspect='auto')
                plt.savefig(gm_img_path, dpi=20)

            if gm_img_path not in results_register:
                results_register.add(gm_img_path)
                # L1Distance.update(l1distance(output_seq[frame_idx].detach(), label_seq[frame_idx]/255))
                # # SSIM.update(ssim_calculate(gm_img, gt_img))

        if para.video:
            logger('seq {} video result generating ...'.format(seq))
            marks = ['Input', para.model, 'GT']
            path = dirname(save_dir)
            frame_start = para.past_frames
            frame_end = seq_length - para.future_frames
            img2video(path=path, size=(3 * W, 1 * H), seq=seq, frame_start=frame_start, frame_end=frame_end,
                      marks=marks, fps=10)

    # logger('Test images : {}'.format(PSNR.count), prefix='\n')
    # logger('Test PSNR : {}'.format(PSNR.avg))
    # logger('Test SSIM : {}'.format(SSIM.avg))
    logger('Average time per image: {}'.format(timer.avg))


def _test_lmdb5(para, logger, model, ds_type, ds_name):
    L1Distance = AverageMeter()
    metrics_name = para.metrics
    module = import_module('train.metrics')
    # metrics = getattr(module, metrics_name)(para).cuda()
    # SSIM = AverageMeter()
    timer = AverageMeter()
    results_register = set()
    B, H, W, C = 1, 180, 240, 1
    
    path = join(para.data_root, ds_name)
    data_test_path = join(path, '{}_{}_blurry'.format(ds_name, ds_type))
    data_test_gt_path = join(path, '{}_{}_gt'.format(ds_name, ds_type))
    data_test_evt_path = join(path, '{}_{}_sbt'.format(ds_name, ds_type))
    env_blur = lmdb.open(data_test_path, map_size=1099511627776)
    env_gt = lmdb.open(data_test_gt_path, map_size=1099511627776)
    env_evt = lmdb.open(data_test_evt_path, map_size=1099511627776)
    txn_blur = env_blur.begin()
    txn_gt = env_gt.begin()
    txn_evt = env_evt.begin()

    data_test_info_path = join(path, '{}_info_{}.pkl'.format(ds_name, ds_type))
    with open(data_test_info_path, 'rb') as f:
        seqs_info = pickle.load(f)

    for seq_idx in range(seqs_info['num']):#
        # if seq_idx!=0 and seq_idx!=1 and seq_idx!=5:
        #     continue
        seq_length = seqs_info[seq_idx]['length']
        seq = '{:03d}'.format(seq_idx)
        logger('seq {} image results generating ...'.format(seq))
        dir_name = '_'.join((para.dataset, para.model, 'test'))
        save_dir = join(para.test_save_dir, dir_name, seq)
        os.makedirs(save_dir, exist_ok=True)
        start = 0
        end = para.test_frames

        input_seq = []
        label_seq = []
        event_seq = []
        label_seq_forinput = []
        for frame_idx in range(start, end):
            code = '%03d_%08d' % (seq_idx, frame_idx)
            code = code.encode()
            blur_img = txn_blur.get(code)
            blur_img = np.frombuffer(blur_img, dtype='uint8')
            blur_img = blur_img.reshape(H, W)[np.newaxis,np.newaxis,:,:].astype(np.float32)
            gt_img = txn_gt.get(code)
            gt_img = np.frombuffer(gt_img, dtype='uint8')
            gt_img = gt_img.reshape(4, H, W)[np.newaxis,:,:,:]
            event_stack = txn_evt.get(code)
            event_stack = np.frombuffer(event_stack, dtype='int8')
            event_stack = event_stack.reshape(16, H, W)[np.newaxis,:].astype(np.float32)

            event_seq.append(event_stack)
            input_seq.append(blur_img)
            label_seq.append(gt_img)
        input_seq = np.ascontiguousarray(np.concatenate(input_seq)) #10,c,h,w
        event_seq = np.ascontiguousarray(np.concatenate(event_seq))#10,4,h,w
        label_seq = np.ascontiguousarray(np.concatenate(label_seq))
        model.eval()
        with torch.no_grad():
            input_seq = torch.from_numpy(input_seq).cuda().unsqueeze(0)/255
            event_seq = torch.from_numpy(event_seq).cuda().unsqueeze(0)/255
        
            # print('input_seq',input_seq.max(),input_seq.min(),input_seq.mean())
            # print('event_seq',event_seq.max(),event_seq.min(),event_seq.mean())
            time_start = time.time()
            output_seq = model([torch.cat([input_seq, event_seq],dim=2), ])#[(1,10,48,h,w)]
            if isinstance(output_seq, (list, tuple)):
                output_seq = output_seq[0]
            timer.update((time.time() - time_start) / len(output_seq), n=len(output_seq))
        output_seq = output_seq.squeeze()
        fm,c,h,w = output_seq.shape
        output_seq = output_seq.reshape(fm, 4,1,h,w)
        for frame_idx in range(para.test_frames):
            for j in range(4):
                gt_img = label_seq[frame_idx,j]
                gt_img = torch.from_numpy(gt_img.astype(np.float32)/255).unsqueeze(0).cuda()
                with torch.no_grad():
                    G = Mgrad(gt_img, para.epsilon, ifmog=False)#(1,6,h,w)
                    G0 = torch.sqrt(torch.pow(G[:,0],2)+torch.pow(G[:,1],2))
                    G1 = torch.sqrt(torch.pow(G[:,2],2)+torch.pow(G[:,3],2))
                    G2 = torch.sqrt(torch.pow(G[:,4],2)+torch.pow(G[:,5],2))
                    G_m = torch.sqrt((torch.pow(G0,2)+torch.pow(G1,2)+torch.pow(G2,2))/3)
                    gt_img = G_m.squeeze().cpu().numpy()
                    gt_mat = G.squeeze().cpu().numpy()
                    gt_img = (gt_img*255/gt_img.max()).astype(np.uint8)
                gt_img_path = join(save_dir, '{:08d}_gt_{}.png'.format(frame_idx + start, j))
                gt_mat_path = join(save_dir, '{:08d}_gt_{}.npy'.format(frame_idx + start, j))
#                 print('output_seq shape:', output_seq.shape)
                g_m = output_seq[frame_idx,j].detach()
                G0 = torch.sqrt(torch.pow(g_m[0],2)+torch.pow(g_m[1],2))
                G1 = torch.sqrt(torch.pow(g_m[2],2)+torch.pow(g_m[3],2))
                G2 = torch.sqrt(torch.pow(g_m[4],2)+torch.pow(g_m[5],2))
                G = torch.sqrt((torch.pow(G0,2)+torch.pow(G1,2)+torch.pow(G2,2))/3)
                gm_img = G.squeeze().cpu().numpy()
                gm_mat = g_m.squeeze().cpu().numpy()
                gm_img = (gm_img*255/gm_img.max()).astype(np.uint8)
                gm_img_path = join(save_dir, '{:08d}_{}_{}.png'.format(frame_idx + start, j, para.model.lower()))
                gm_mat_path = join(save_dir, '{:08d}_{}_{}.npy'.format(frame_idx + start, j, para.model.lower()))
                # print('deblur_img,',deblur_img.max(),deblur_img.min(),deblur_img.mean())
                # print('gt_img',gt_img.max(),gt_img.min(),gt_img.mean())
                cv2.imwrite(gt_img_path, gt_img)
                cv2.imwrite(gm_img_path, gm_img)
                np.save(gt_mat_path, gt_mat)
                np.save(gm_mat_path, gm_mat)

            if gm_img_path not in results_register:
                results_register.add(gm_img_path)
                # L1Distance.update(l1distance(output_seq[frame_idx].detach(), label_seq[frame_idx]/255))
                # # SSIM.update(ssim_calculate(gm_img, gt_img))

        if para.video:
            logger('seq {} video result generating ...'.format(seq))
            marks = ['Input', para.model, 'GT']
            path = dirname(save_dir)
            frame_start = para.past_frames
            frame_end = seq_length - para.future_frames
            img2video(path=path, size=(3 * W, 1 * H), seq=seq, frame_start=frame_start, frame_end=frame_end,
                      marks=marks, fps=10)

    # logger('Test images : {}'.format(PSNR.count), prefix='\n')
    # logger('Test PSNR : {}'.format(PSNR.avg))
    # logger('Test SSIM : {}'.format(SSIM.avg))
    logger('Average time per image: {}'.format(timer.avg))


def _test_lmdb(para, logger, model, ds_type, ds_name):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    timer = AverageMeter()
    results_register = set()
    B, H, W, C = 1, 180, 240, 1
    
    path = join(para.data_root, ds_name)
    data_test_path = join(path, '{}_{}_blurry'.format(ds_name, ds_type))
    data_test_gt_path = join(path, '{}_{}_gt'.format(ds_name, ds_type))
    data_test_sbt48c_path = join(path, '{}_{}_sbt'.format(ds_name, ds_type))

    env_sbt48c = lmdb.open(data_test_sbt48c_path, map_size=1099511627776)
    txn_sbt48c = env_sbt48c.begin()
    env_blur = lmdb.open(data_test_path, map_size=1099511627776)
    env_gt = lmdb.open(data_test_gt_path, map_size=1099511627776)
    txn_blur = env_blur.begin()
    txn_gt = env_gt.begin()

    data_test_info_path = join(path, '{}_info_{}.pkl'.format(ds_name, ds_type))
    with open(data_test_info_path, 'rb') as f:
        seqs_info = pickle.load(f)

    for seq_idx in range(seqs_info['num']):#seqs_info['num']
        # if seq_idx!=0 and seq_idx!=1 and seq_idx!=5:
        #     continue
        seq_length = seqs_info[seq_idx]['length']
        seq = '{:03d}'.format(seq_idx)
        logger('seq {} image results generating ...'.format(seq))
        dir_name = '_'.join((para.dataset, para.model, 'test'))
        save_dir = join(para.test_save_dir, dir_name, seq)
        os.makedirs(save_dir, exist_ok=True)
        start = 0
        end = para.test_frames

        input_seq = []
        label_seq = []
        event_seq = []
        for frame_idx in range(start, end):
            code = '%03d_%08d' % (seq_idx, frame_idx)
            code = code.encode()
            blur_img = txn_blur.get(code)
            blur_img = np.frombuffer(blur_img, dtype='uint8')
            blur_img = blur_img.reshape(H, W)[np.newaxis,np.newaxis,:,:].astype(np.float32)
            gt_img = txn_gt.get(code)
            gt_img = np.frombuffer(gt_img, dtype='uint8')
            gt_img = gt_img.reshape(4, H, W)[np.newaxis,:,:,:]
            event_stack = txn_sbt48c.get(code)
            event_stack = np.frombuffer(event_stack, dtype='int8')
            event_stack = event_stack.reshape(16, H, W)[np.newaxis,:].astype(np.float32)
        
            event_seq.append(event_stack)
            input_seq.append(blur_img)
            label_seq.append(gt_img)
        input_seq = np.ascontiguousarray(np.concatenate(input_seq)) #10,c,h,w
        event_seq = np.ascontiguousarray(np.concatenate(event_seq))#10,48,h,w
        model.eval()
        with torch.no_grad():
            input_seq = torch.from_numpy(input_seq).cuda()/255
            event_seq = torch.from_numpy(event_seq).cuda()/255
            # print('input_seq',input_seq.max(),input_seq.min(),input_seq.mean())
            # print('event_seq',event_seq.max(),event_seq.min(),event_seq.mean())
            time_start = time.time()
            output_seq = model([torch.cat([input_seq,event_seq], dim=1).unsqueeze(0), ])#[(1,10,7,h,w)]
            if isinstance(output_seq, (list, tuple)):
                output_seq = output_seq[0]
            timer.update((time.time() - time_start) / len(output_seq), n=len(output_seq))
        dbout = output_seq.squeeze()
        gt_imgs = []
        deblur_imgs = []
        # print("out shape, input shape, gt leng: {}, {}, {}".format(output_seq.shape, torch.cat([input_seq, event_seq], dim=1).unsqueeze(0).shape, len(label_seq)))
        for frame_idx in range(para.past_frames*2, para.test_frames - para.future_frames*2):
            
            blur_img = input_seq.squeeze()[frame_idx][:,:]
#             print(blur_img.shape)
            blur_img = normalize_reverse(blur_img, centralize=para.centralize, normalize=para.normalize)
            blur_img = blur_img.detach().cpu().numpy().astype(np.uint8)
            blur_img_path = join(save_dir, '{:08d}_input.png'.format(frame_idx + start))
            # blur_img = blur_img.transpose((1,2,0))[:,:,[2,1,0]]
#             print("blur_img, tranpose",blur_img.shape)
            cv2.imwrite(blur_img_path, blur_img)
            
            for j in range(4):
                gt_img = label_seq[frame_idx].squeeze()[j]
                gt_img_path = join(save_dir, '{:08d}_gt_{}.png'.format(frame_idx + start, j))
#                 print('output_seq shape:', output_seq.shape)
                deblur_img = dbout[frame_idx-para.past_frames*2,j]
                deblur_img = deblur_img.detach().cpu().numpy()
                deblur_img = np.clip(deblur_img, 0, 255).astype(np.uint8)
                deblur_img_path = join(save_dir, '{:08d}_{}_{}.png'.format(frame_idx + start, j, para.model.lower()))
                cv2.imwrite(gt_img_path, gt_img)
                gt_imgs.append(gt_img[np.newaxis,:])
                cv2.imwrite(deblur_img_path, deblur_img)
                deblur_imgs.append(deblur_img[np.newaxis,:])

        gt_imgs = np.concatenate(gt_imgs)
        deblur_imgs = np.concatenate(deblur_imgs)
        print('deblur_imgs shape, gt shape: {}, {}'.format(deblur_imgs.shape, gt_imgs.shape))
        PSNR.update(psnr_calculate(deblur_imgs, gt_imgs))
        SSIM.update(ssim_calculate(deblur_imgs, gt_imgs))

        if para.video:
            logger('seq {} video result generating ...'.format(seq))
            marks = ['Input', para.model, 'GT']
            path = dirname(save_dir)
            frame_start = para.past_frames
            frame_end = seq_length - para.future_frames
            img2video(path=path, size=(3 * W, 1 * H), seq=seq, frame_start=frame_start, frame_end=frame_end,
                      marks=marks, fps=10)

    logger('Test images : {}'.format(PSNR.count), prefix='\n')
    logger('Test PSNR : {}'.format(PSNR.avg))
    logger('Test SSIM : {}'.format(SSIM.avg))
    logger('Average time per image: {}'.format(timer.avg))

def _test_lmdb_cdvd(para, logger, model, ds_type, ds_name):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    timer = AverageMeter()
    results_register = set()
    B, H, W, C = 1, 180, 240, 1
    
    path = join(para.data_root, ds_name)
    data_test_path = join(path, '{}_{}_blurry'.format(ds_name, ds_type))
    data_test_gt_path = join(path, '{}_{}_gt'.format(ds_name, ds_type))
    data_test_sbt48c_path = join(path, '{}_{}_sbt'.format(ds_name, ds_type))

    env_sbt48c = lmdb.open(data_test_sbt48c_path, map_size=1099511627776)
    txn_sbt48c = env_sbt48c.begin()
    env_blur = lmdb.open(data_test_path, map_size=1099511627776)
    env_gt = lmdb.open(data_test_gt_path, map_size=1099511627776)
    txn_blur = env_blur.begin()
    txn_gt = env_gt.begin()

    data_test_info_path = join(path, '{}_info_{}.pkl'.format(ds_name, ds_type))
    with open(data_test_info_path, 'rb') as f:
        seqs_info = pickle.load(f)

    for seq_idx in range(seqs_info['num']):#seqs_info['num']
        # if seq_idx!=0 and seq_idx!=1 and seq_idx!=5:
        #     continue
        seq_length = seqs_info[seq_idx]['length']
        seq = '{:03d}'.format(seq_idx)
        logger('seq {} image results generating ...'.format(seq))
        dir_name = '_'.join((para.dataset, para.model, 'test'))
        save_dir = join(para.test_save_dir, dir_name, seq)
        os.makedirs(save_dir, exist_ok=True)
        start = 0
        end = para.test_frames

        input_seqs = []
        label_seqs = []
        event_seqs = []
        for frame_idx in range(start, end):
            event_seq = []
            label_seq = []
            input_seq = []
            for frame_idx_jj in range(5):
                code = '%03d_%08d' % (seq_idx, frame_idx+frame_idx_jj)
                code = code.encode()
                blur_img = txn_blur.get(code)
                blur_img = np.frombuffer(blur_img, dtype='uint8')
                blur_img = blur_img.reshape(H, W)[np.newaxis,np.newaxis,:,:].astype(np.float32)
                gt_img = txn_gt.get(code)
                gt_img = np.frombuffer(gt_img, dtype='uint8')
                gt_img = gt_img.reshape(4, H, W)[np.newaxis,:,:,:]
                event_stack = txn_sbt48c.get(code)
                event_stack = np.frombuffer(event_stack, dtype='int8')
                event_stack = event_stack.reshape(16, H, W)[np.newaxis,:].astype(np.float32)

                event_seq.append(event_stack)
                input_seq.append(blur_img)
                label_seq.append(gt_img)
            input_seq = np.ascontiguousarray(np.concatenate(input_seq)) #5,c,h,w
            event_seq = np.ascontiguousarray(np.concatenate(event_seq))#5,16,h,w
            input_seqs.append(input_seq)
            event_seqs.append(event_seq)
            label_seqs.append(label_seq)
        model.eval()
        with torch.no_grad():
            outputs = []
            for frame_idx in range(start, end):
                input_seq = input_seqs[frame_idx]
                event_seq = event_seqs[frame_idx]
                input_seq = torch.from_numpy(input_seq).cuda()/255
                event_seq = torch.from_numpy(event_seq).cuda()/255
                recons_1, recons_2, recons_3, recons_2_iter = model([torch.cat([input_seq,event_seq], dim=1).unsqueeze(0), ])#[(1,10,7,h,w)]
                outputs.append(recons_2_iter.unsqueeze(2))
        dbout = torch.cat(outputs, dim=1).squeeze() # (n,fm,c,h,w)
        gt_imgs = []
        deblur_imgs = []
        # print("out shape, input shape, gt leng: {}, {}, {}".format(output_seq.shape, torch.cat([input_seq, event_seq], dim=1).unsqueeze(0).shape, len(label_seq)))
        for frame_idx in range(para.past_frames, para.test_frames - para.future_frames):
            
            blur_img = input_seqs[frame_idx][2].squeeze()
#             print(blur_img.shape)
            blur_img = normalize_reverse(blur_img, centralize=para.centralize, normalize=para.normalize)
            blur_img = blur_img.astype(np.uint8)
            blur_img_path = join(save_dir, '{:08d}_input.png'.format(frame_idx + start))
            # blur_img = blur_img.transpose((1,2,0))[:,:,[2,1,0]]
#             print("blur_img, tranpose",blur_img.shape)
            # import pdb; pdb.set_trace()
            cv2.imwrite(blur_img_path, blur_img)
            
            for j in range(1):
                gt_img = label_seqs[frame_idx][2].squeeze()[j]
                gt_img_path = join(save_dir, '{:08d}_gt_{}.png'.format(frame_idx + start, j))
#                 print('output_seq shape:', output_seq.shape)
                deblur_img = dbout[frame_idx-para.past_frames]
                deblur_img = deblur_img.detach().cpu().numpy()
                deblur_img = np.clip(deblur_img, 0, 255).astype(np.uint8)
                deblur_img_path = join(save_dir, '{:08d}_{}_{}.png'.format(frame_idx + start, j, para.model.lower()))
                cv2.imwrite(gt_img_path, gt_img)
                gt_imgs.append(gt_img[np.newaxis,:])
                cv2.imwrite(deblur_img_path, deblur_img)
                deblur_imgs.append(deblur_img[np.newaxis,:])

        gt_imgs = np.concatenate(gt_imgs)
        deblur_imgs = np.concatenate(deblur_imgs)
        print('deblur_imgs shape, gt shape: {}, {}'.format(deblur_imgs.shape, gt_imgs.shape))
        PSNR.update(psnr_calculate(deblur_imgs, gt_imgs))
        SSIM.update(ssim_calculate(deblur_imgs, gt_imgs))

        if para.video:
            logger('seq {} video result generating ...'.format(seq))
            marks = ['Input', para.model, 'GT']
            path = dirname(save_dir)
            frame_start = para.past_frames
            frame_end = seq_length - para.future_frames
            img2video(path=path, size=(3 * W, 1 * H), seq=seq, frame_start=frame_start, frame_end=frame_end,
                      marks=marks, fps=10)

    logger('Test images : {}'.format(PSNR.count), prefix='\n')
    logger('Test PSNR : {}'.format(PSNR.avg))
    logger('Test SSIM : {}'.format(SSIM.avg))
    logger('Average time per image: {}'.format(timer.avg))
