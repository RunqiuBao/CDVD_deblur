import pickle
import random
from os.path import join

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import Crop, Flip, ToTensor, normalize


class DeblurDataset(Dataset): #runqiu: deblur for event frame!
    def __init__(self, path, crop_size=(256, 256), ds_type='train', centralize=True,
                 normalize=True):
        ds_name = 'gopro'
        self.datapath_tsi = join(path, '{}_{}_event_tsi'.format(ds_name, ds_type))
        self.datapath_sbt = join(path, '{}_{}_event_pn'.format(ds_name, ds_type))
        with open(join(path, '{}_info_{}.pkl'.format(ds_name, ds_type)), 'rb') as f:
            self.seqs_info = pickle.load(f)
        self.transform = transforms.Compose([Crop(crop_size),ToTensor()])#, Flip() 
        self.crop_h, self.crop_w = crop_size
        self.W = 1280
        self.H = 720
        self.C = 8
        if ds_type == 'train':
            self.seqlist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        else:
            self.seqlist = [0,1,2,3,4,5,6,7,8,9,10]
        self.normalize = normalize
        self.centralize = centralize
        self.env_tsi = lmdb.open(self.datapath_tsi, map_size=1099511627776)
        self.txn_tsi = self.env_tsi.begin()
        self.env_sbt = lmdb.open(self.datapath_sbt, map_size=1099511627776)
        self.txn_sbt = self.env_sbt.begin()
    
    def get_index(self):
        seq_idx = random.choice(self.seqlist)
        frame_idx = random.randint(0, self.seqs_info[seq_idx]['length']-1)
        
        return seq_idx, frame_idx

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.get_index()

        top = random.randint(0, self.H - self.crop_h)
        left = random.randint(0, self.W - self.crop_w)
        flip_lr_flag = random.randint(0, 1)
        flip_ud_flag = random.randint(0, 1)
        sample = {'top': top, 'left': left, 'flip_lr': flip_lr_flag, 'flip_ud': flip_ud_flag}

        try:
            tsi_stack, sbt_stack = self.get_img(seq_idx, frame_idx, sample)#8,h,w
        except TypeError as err:
            print('Handling run-time error:', err)
            print('failed case: seq_idx {}, frame_idx {}'.format(seq_idx, frame_idx))
        tsi_stack = tsi_stack[[0,4,1,5,2,6,3,7],:,:]
        tsi_stack = tsi_stack[0:2,:,:]
        sbt_stack = sbt_stack[[0,4,1,5,2,6,3,7],:,:]
        sbt_stack = sbt_stack[0:2,:,:]
        inputs = torch.cat([sbt_stack, tsi_stack], dim=0) #2x2,h,w
#         print("in loader inputs.shape: {}".format(inputs.shape))
        return inputs

    def get_img(self, seq_idx, frame_idx, sample):
        code = '%03d_%08d' % (seq_idx, frame_idx)
        code = code.encode()
        tsi_img = self.txn_tsi.get(code)
        tsi_img = np.frombuffer(tsi_img, dtype='uint8')
        tsi_img = tsi_img.reshape(8, self.H, self.W)
        sbt_img = self.txn_sbt.get(code)
        sbt_img = np.frombuffer(sbt_img, dtype='int8')
        sbt_img = sbt_img.reshape(8, self.H, self.W)

        sample['event'] = sbt_img
        sample['tsi'] = tsi_img
        sample['iseventfocus'] = True
        sample = self.transform(sample)
        
        sbt_img = normalize(sample['event'], iseventfocus=True, istsi=False)
        tsi_img = normalize(sample['tsi'], iseventfocus=True, istsi=True)
#         print("tsi image max: {}".format(tsi_img.max()))
        return tsi_img.squeeze(), sbt_img.squeeze()
    
    def __len__(self):
        return self.seqs_info['length']


class Dataloader:
    def __init__(self, para, device_id, ds_type='train'):
        path = join(para.data_root, para.dataset)
        dataset = DeblurDataset(path, para.patch_size, ds_type, para.centralize, para.normalize)
        gpus = para.num_gpus
        bs = para.batch_size
        ds_len = len(dataset)
        if para.trainer_mode == 'ddp' or 'ddp_flow':
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
                sampler=sampler
            )
            loader_len = np.ceil(ds_len / gpus)
            self.loader_len = int(np.ceil(loader_len / bs) * bs)

        elif para.trainer_mode == 'dp':
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=True,
                num_workers=para.threads,
                pin_memory=True
            )
            self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len
