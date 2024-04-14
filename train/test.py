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
from .metrics import psnr_calculate, ssim_calculate, mog_calculate, average_timestamp_calculate, ssim_calc_torch
from .utils import AverageMeter, img2video

from.utils import flow_to_image, flow_uv_to_colors, make_colorwheel
import copy
from train.utils import Mgrad, Mgrad_mono
from importlib import import_module

import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.testconfig import TestConfig
datasetInfo = TestConfig.get_cnf()['test_param']


def TestTiming(para, logger):
    model = Model(para).cuda()
    model = nn.DataParallel(model)
    model.eval()
    if para.model[:4] == 'cdvd':
        inputTensor = torch.rand(1, 5, 3, 720, 1280)
        inputLabels = torch.rand(1, 5, 4, 3, 720, 1280)
    with torch.no_grad():
        starttime = time.time()
        outputTensor = model([inputTensor, inputLabels])[0]
        print("inference time: {} sec".format((time.time() - starttime) * 4 / 9))
    from IPython import embed; print('here!'); embed()


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

def _test_lmdb_cdvd(para, logger, model, ds_type, ds_name):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    timer = AverageMeter()
    results_register = set()
    H, W, C, numGt, Cev = datasetInfo['H'], datasetInfo['W'], datasetInfo['Cimg'], datasetInfo['numGt'], datasetInfo['Cev']
    if datasetInfo['datasetName'] == 'beamsplitter':
        cropMargin = datasetInfo['cropMargin']
    
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
        start = seqs_info[seq_idx]['start']
        end = seqs_info[seq_idx]['end']
        seq_idx += seqs_info['seq_index_shift']
        seq = '{:03d}'.format(seq_idx)
        logger('seq {} image results generating ...'.format(seq))
        dir_name = '_'.join((para.dataset, para.model, 'test'))
        save_input_dir = join(para.test_save_dir, dir_name, seq, 'input')
        save_gt_dir = join(para.test_save_dir, dir_name, seq, 'gt')
        save_output_dir = join(para.test_save_dir, dir_name, seq, 'output')
        os.makedirs(save_input_dir, exist_ok=True)
        os.makedirs(save_gt_dir, exist_ok=True)
        os.makedirs(save_output_dir, exist_ok=True)

        input_seqs = []
        label_seqs = []
        event_seqs = []
        pbar = tqdm(total=seq_length, ncols=80)
        for frame_idx in range(start, end):
            event_seq = []
            label_seq = []
            input_seq = []
            for frame_idx_jj in range(5):
                if datasetInfo['datasetName'] == 'beamsplitter':
                    code = '%03d_%06d' % (seq_idx, frame_idx + frame_idx_jj)
                    code = code.encode()
                    blur_img = txn_blur.get(code)
                    blur_img = np.frombuffer(blur_img, dtype='uint8')
                    blur_img = blur_img.reshape(H, W, C)
                    blur_img = np.einsum('ijk->kij', blur_img)[np.newaxis, :, :, cropMargin:W - cropMargin].astype(np.float32)  # (1, 3, H, W)
                    gt_img = txn_gt.get(code)
                    gt_img = np.frombuffer(gt_img, dtype='uint8')
                    gt_img = gt_img.reshape(numGt, H, W, C)
                    gt_img = np.einsum('ijkl->iljk', gt_img)[np.newaxis, :, :, :, cropMargin:W - cropMargin]  # (1, 4, 3, H, W)
                    event_stack = txn_sbt48c.get(code)
                    event_stack = np.frombuffer(event_stack, dtype='int8')
                    event_stack = event_stack.reshape(Cev, H, W)[np.newaxis, :, :, cropMargin:W - cropMargin].astype(np.float32)  # (1, 16, H, W)
                else:
                    code = '%03d_%08d' % (seq_idx, frame_idx + frame_idx_jj)
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
                if frame_idx_jj == 2:
                    label_seqs.append(gt_img[:, 2, ...][:, np.newaxis])
            input_seq = np.ascontiguousarray(np.concatenate(input_seq)) #5,c,h,w
            event_seq = np.ascontiguousarray(np.concatenate(event_seq))#5,16,h,w
            input_seqs.append(input_seq)
            event_seqs.append(event_seq)
            pbar.update(1)
        pbar.close()
        model.eval()
        numInference = seq_length // (para.frames)
        if datasetInfo['datasetName'] == 'beamsplitter':
            labelShape = (1, para.frames, datasetInfo['Cimg'], datasetInfo['H'], datasetInfo['W'] - cropMargin * 2)
            dboutShape = (1, para.frames, datasetInfo['Cimg'], datasetInfo['H'], datasetInfo['W'] - cropMargin * 2)
        pbar = tqdm(total=numInference, ncols=80)
        outputs = []
        for indexInference in range(numInference):
            with torch.no_grad():
                for frame_idx in range(indexInference * para.frames, (indexInference + 1) * para.frames):
                    input_seq = input_seqs[frame_idx]
                    event_seq = event_seqs[frame_idx]
                    input_seq = torch.from_numpy(input_seq).cuda() / 255
                    event_seq = torch.from_numpy(event_seq).cuda() / 255
                    recons_1, recons_2, recons_3, recons_2_iter = model([torch.cat([input_seq, event_seq], dim=1).unsqueeze(0), ])#[(1,10,7,h,w)]
                    outputs.append(recons_2_iter.unsqueeze(1))
            pbar.update(1)
        pbar.close()
        
        dbout = torch.cat(outputs, dim=1).squeeze() # (n,fm,c,h,w)
        dbout = dbout.squeeze(0)
        label_seq_output = torch.from_numpy(np.concatenate(label_seqs, axis=1)).squeeze(0)
        deblur_imgs = []
        gt_imgs = []
        psnrThisSeq = None
        ssimThisSeq = None
        frameCount = 0
        logger('start save.')        
        for frame_idx in range(dbout.shape[0]):
            blur_img = input_seqs[frame_idx].squeeze()[2]
            blur_img = blur_img.astype(np.uint8)
            if datasetInfo['datasetName'] == "beamsplitter":
                blur_img = np.einsum('ijk->jki', blur_img)
            blur_img_path = join(save_input_dir, '{:06d}.png'.format(frame_idx))            
            cv2.imwrite(blur_img_path, blur_img)

            gt_img = label_seq_output[frame_idx].squeeze().numpy()
            gt_img_path = join(save_gt_dir, '{:06d}.png'.format(frame_idx))
            deblur_img = dbout[frame_idx].detach().cpu().numpy()
            deblur_img = np.clip(deblur_img, 0, 255).astype(np.uint8)
            deblur_img_path = join(save_output_dir, '{:06d}.png'.format(frame_idx))
            if datasetInfo['datasetName'] == "beamsplitter":
                gt_img = np.einsum('ijk->jki', gt_img)
                deblur_img = np.einsum('ijk->jki', deblur_img)
            cv2.imwrite(gt_img_path, gt_img.squeeze())
            gt_imgs.append(gt_img[np.newaxis, :])
            cv2.imwrite(deblur_img_path, deblur_img.squeeze())
            deblur_imgs.append(deblur_img[np.newaxis, :])

            psnrOneFrame = psnr_calculate(gt_img, deblur_img)
            psnrThisSeq = psnrOneFrame if psnrThisSeq == None else (psnrThisSeq + psnrOneFrame)
            ssimOneFrame = ssim_calc_torch(dbout[frame_idx].detach().cpu().clamp(0, 255.0).round().unsqueeze(0).to(torch.float), label_seq_output[frame_idx].unsqueeze(0).to(torch.float))
            ssimThisSeq = ssimOneFrame if ssimThisSeq == None else (ssimThisSeq + ssimOneFrame)
            frameCount += 1
        if not para.model == 'IGP2':
            logger('seq {}, mean psnr: {}, mean ssim: {}'.format(seq_idx, psnrThisSeq / frameCount, ssimThisSeq / frameCount))
            PSNR.update(psnrThisSeq / frameCount, frameCount)
            SSIM.update(ssimThisSeq / frameCount, frameCount)
    if not para.model == 'IGP2':
        logger('Test images : {}, Average PSNR {}, Average SSIM {}'.format(PSNR.count, PSNR.avg, SSIM.avg), prefix='\n')
    logger('Average time per image: {}'.format(timer.avg))

