import pickle
import random
from os.path import join
from cv2 import blur

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import Crop, Flip, ToTensor, normalize


class DeblurDataset(Dataset):
    def __init__(self, path, frames, future_frames, past_frames, crop_size=(256, 256), ds_type='train', centralize=True,
                 normalize=True, para=None):
        ds_name = 'beamsplitter'
        self.para = para
        self.datapath_blur = join(path, '{}_{}_blurry'.format(ds_name, ds_type))
        self.datapath_gt = join(path, '{}_{}_gt'.format(ds_name, ds_type))
        self.datapath_event2 = join(path, '{}_{}_sbt'.format(ds_name, ds_type))

        with open(join(path, '{}_info_{}.pkl'.format(ds_name, ds_type)), 'rb') as f:
            self.seqs_info = pickle.load(f)
        self.ds_type = ds_type
        if ds_type=='train' or 'valid':
            self.transform = transforms.Compose([Crop(crop_size), ToTensor()])#, Flip()
        else:
            self.transform = transforms.Compose([ToTensor()]) 
        self.frames = frames
        self.crop_h, self.crop_w = crop_size
        self.W = 1280
        self.H = 720
        self.C = 3
        self.num_ff = future_frames
        self.num_pf = past_frames
        if ds_type == 'train':
            self.seqlist = [0,1,2,3]
        else:
            self.seqlist = [0,1]
        self.normalize = normalize
        self.centralize = centralize
        self.env_blur = lmdb.open(self.datapath_blur, map_size=1099511627776)
        self.env_gt = lmdb.open(self.datapath_gt, map_size=1099511627776)
        # self.env_event = lmdb.open(self.datapath_event, map_size=1099511627776)
        self.env_event2 = lmdb.open(self.datapath_event2, map_size=1099511627776)
        self.txn_blur = self.env_blur.begin()
        self.txn_gt = self.env_gt.begin()
        # self.txn_event = self.env_event.begin()
        self.txn_event2 = self.env_event2.begin()
    
    def get_index(self):
        seq_idx = random.choice(self.seqlist)
        frame_idx = random.randint(0, self.seqs_info[seq_idx]['length'] - self.frames)
        
        return seq_idx, frame_idx

    def __getitem__(self, idx):
        idx += 1
        ori_idx = idx
        seq_idx, frame_idx = 0, 0
        blur_imgs, sharp_imgs, event_stacks = list(), list(), list()
        
        for i in range(self.seqs_info['num']):
            seq_length = self.seqs_info[i]['length'] - self.frames + 1
            if idx - seq_length <= 0:
                seq_idx = i + self.seqs_info['seq_index_shift']
                frame_idx = self.seqs_info[i]['start'] + idx - 1
                break
            else:
                idx -= seq_length

        top = random.randint(0, self.H - self.crop_h)
        left = random.randint(300, self.W - 300 - self.crop_w)
        flip_lr_flag = random.randint(0, 1)
        flip_ud_flag = random.randint(0, 1)
        sample = {'top': top, 'left': left, 'flip_lr': flip_lr_flag, 'flip_ud': flip_ud_flag}

        for i in range(self.frames):
            try:
                blur_img, sharp_img, event = self.get_img(seq_idx, frame_idx + i, sample)
                blur_imgs.append(blur_img)
                sharp_imgs.append(sharp_img)
                event_stacks.append(event)
            except TypeError as err:
                print('Handling run-time error:', err)
                print('failed case: idx {}, seq_idx {}, frame_idx {}'.format(ori_idx, seq_idx, frame_idx))
        blur_imgs = torch.cat(blur_imgs, dim=0)
        sharp_imgs = torch.cat(sharp_imgs, dim=0)
        event_stacks = torch.cat(event_stacks, dim=0)
        inputs = torch.cat([blur_imgs, event_stacks], dim=1)
        # inputs = inputs[:,:,:256, :320]
        # sharp_imgs = sharp_imgs[:,:, :,:256,:320]
        return inputs, sharp_imgs, seq_idx

    def get_img(self, seq_idx, frame_idx, sample):
        if seq_idx == 2 and self.ds_type == 'train':
            seq_idx = 3
        code = '%03d_%06d' % (seq_idx, frame_idx)
        code = code.encode()
        blur_img = self.txn_blur.get(code)
        blur_img = np.frombuffer(blur_img, dtype='uint8')
        blur_img = blur_img.reshape(self.H, self.W, self.C)
        blur_img = np.einsum('ijk->kij', blur_img)
        sharp_img = self.txn_gt.get(code)
        sharp_img = np.frombuffer(sharp_img, dtype='uint8')
        sharp_img = sharp_img.reshape(4, self.H, self.W, self.C)
        sharp_img = np.einsum('ijkl->iljk', sharp_img)
        event2_stack = self.txn_event2.get(code)
        event2_stack = np.frombuffer(event2_stack, dtype='int8')
        event2_stack = event2_stack.reshape(16, self.H, self.W)

        sample['image'] = blur_img
        sample['label'] = sharp_img
        sample['event'] = np.array(event2_stack)
        sample = self.transform(sample)
        blur_img = normalize(sample['image'], centralize=self.centralize, normalize=self.normalize)
        # FIXME: 
        if self.para.model == 'IGP2':
            sharp_img = normalize(sample['label'], centralize=self.centralize, normalize=self.normalize)  #igp
        else:
            sharp_img = sample['label'] #deblur
        event2_stack = normalize(sample['event'], centralize=self.centralize, normalize=self.normalize)

        return blur_img.unsqueeze(0), sharp_img.unsqueeze(0), event2_stack.unsqueeze(0)

    def __len__(self):
        return self.seqs_info['length'] - (self.frames - 1) * self.seqs_info['num']


class Dataloader:
    def __init__(self, para, device_id, ds_type='train'):
        path = join(para.data_root, para.dataset)
        dataset = DeblurDataset(path, para.frames, para.future_frames, para.past_frames, para.patch_size, ds_type,
                                para.centralize, para.normalize, para=para)
        gpus = para.num_gpus
        bs = para.batch_size
        ds_len = len(dataset)
        if para.trainer_mode == 'ddp':
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=para.num_gpus,
                rank=device_id
            )
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=False,
                num_workers=para.threads,
                pin_memory=True,
                sampler=sampler,
            )
            loader_len = np.ceil(ds_len / gpus)
            self.loader_len = int(np.ceil(loader_len / bs) * bs)

        elif para.trainer_mode == 'dp':
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=False,
                num_workers=para.threads,
                pin_memory=True,
            )
            self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len
