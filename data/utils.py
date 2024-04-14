import numpy as np
import torch


class Crop(object):
    """
    Crop randomly the image in a sample.
    Args: output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label, event = sample['image'], sample['label'], sample['event']
        top, left = sample['top'], sample['left']
        new_h, new_w = self.output_size
        sample['image'] = image[..., top: top + new_h,
                          left: left + new_w]
        sample['label'] = label[..., top: top + new_h,
                          left: left + new_w]
        sample['event'] = event[..., top: top + new_h,
                          left: left + new_w]
        return sample


class Flip(object):
    """
    shape is (h,w,c)
    """

    def __call__(self, sample):
        flag_lr = sample['flip_lr']
        flag_ud = sample['flip_ud']
        if flag_lr == 1:
            sample['image'] = np.fliplr(sample['image'])
            sample['label'] = np.fliplr(sample['label'])
            sample['event'] = np.fliplr(sample['event'])
        if flag_ud == 1:
            sample['image'] = np.flipud(sample['image'])
            sample['label'] = np.flipud(sample['label'])
            sample['event'] = np.fliplr(sample['event'])

        return sample


class Rotate(object):
    """
    shape is (h,w,c)
    """

    def __call__(self, sample):
        flag = sample['rotate']
        if flag == 1:
            sample['image'] = sample['image'].transpose(1, 0, 2)
            sample['label'] = sample['label'].transpose(1, 0, 2)

        return sample


class Sharp2Sharp(object):
    def __call__(self, sample):
        flag = sample['s2s']
        if flag < 1:
            sample['image'] = sample['label'].copy()
        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        image, label, event = sample['image'], sample['label'], sample['event']
        image = np.ascontiguousarray(image).astype(np.float32)
        label = np.ascontiguousarray(label).astype(np.float32)
        event = np.ascontiguousarray(event).astype(np.float32)
        sample['image'] = torch.from_numpy(image)
        sample['label'] = torch.from_numpy(label)
        sample['event'] = torch.from_numpy(event)
        return sample


def normalize(x, centralize=False, normalize=True, val_range=255.0):
    if centralize:
        x = x - val_range / 2
    if normalize:
        x = x / val_range

    return x

def normalize_reverse(x, centralize=False, normalize=False, val_range=255.0):
    if normalize:
        x = x * val_range

    return x

from os.path import join
def savetensor(x, save_dir):
    path = join(save_dir, 'sample.npy')
    with open(path, 'wb') as f:
        np.save(f, x)
