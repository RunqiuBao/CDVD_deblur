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
        if sample['iseventfocus']:
            events = sample['event']
            tsi = sample['tsi']
            top, left = sample['top'], sample['left']
            new_h, new_w = self.output_size
            sample['event'] = events[:, top: top + new_h, left: left + new_w]
            sample['tsi'] = tsi[:, top: top + new_h, left: left + new_w]
        elif 'isdeblurandframeinterpolation2' in sample.keys():
            image, label = sample['image'], sample['label']
            top, left = sample['top'], sample['left']
            new_h, new_w = self.output_size
            sample['image'] = image[:,top: top + new_h,
                              left: left + new_w]
            sample['label'] = label[:,:,top: top + new_h,
                              left: left + new_w]
        elif 'isdeblurandframeinterpolation' in sample.keys():
            image, label, events = sample['image'], sample['label'], sample['event']
            top, left = sample['top'], sample['left']
            new_h, new_w = self.output_size
            sample['image'] = image[:,top: top + new_h,
                              left: left + new_w]
            sample['label'] = label[:,:,top: top + new_h,
                              left: left + new_w]
            sample['event'] = events[:,top: top + new_h,
                              left: left + new_w]

        elif 'ismodel7' in sample.keys():
            image, label, events = sample['image'], sample['label'], sample['event']
            top, left = sample['top'], sample['left']
            new_h, new_w = self.output_size
            sample['image'] = image[:,top: top + new_h,
                              left: left + new_w]
            sample['label'] = label[:,:,top: top + new_h,
                              left: left + new_w]
            sample['event'] = events[:,top//2: top//2 + new_h//2,
                              left//2: left//2 + new_w//2]
        else:
            image, label, events = sample['image'], sample['label'], sample['event']
            top, left = sample['top'], sample['left']
            new_h, new_w = self.output_size
            sample['image'] = image[top: top + new_h,
                              left: left + new_w]
            sample['label'] = label[top: top + new_h,
                              left: left + new_w]
            sample['event'] = events[top: top + new_h,
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
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if 'iseventfocus' in sample.keys():
            event = sample['event']
            tsi = sample['tsi']
            event = np.ascontiguousarray(event)#8,h,w
            tsi = np.ascontiguousarray(tsi)#8,h,w
            sample['event'] = torch.from_numpy(event).float()
            sample['tsi'] = torch.from_numpy(tsi).float()
        elif 'isdeblurandframeinterpolation2' in sample.keys():
            image, label = sample['image'], sample['label']
            image = np.ascontiguousarray(image[np.newaxis, :])#1,3,h,w
            label = np.ascontiguousarray(label[np.newaxis, :])#1,4,3,h,w
            sample['image'] = torch.from_numpy(image).float()
            sample['label'] = torch.from_numpy(label).float()
        elif 'isdeblurandframeinterpolation' in sample.keys():
            image, label, event = sample['image'], sample['label'], sample['event']
            image = np.ascontiguousarray(image[np.newaxis, :])#1,3,h,w
            label = np.ascontiguousarray(label[np.newaxis, :])#1,4,3,h,w
            event = np.ascontiguousarray(event[np.newaxis, :])#1,4,h,w
            sample['image'] = torch.from_numpy(image).float()
            sample['label'] = torch.from_numpy(label).float()
            sample['event'] = torch.from_numpy(event).float()
        elif 'ismodel7' in sample.keys():
            image, label, event = sample['image'], sample['label'], sample['event']
            image = np.ascontiguousarray(image[np.newaxis, :])#1,3,h,w
            label = np.ascontiguousarray(label[np.newaxis, :])#1,4,3,h,w
            event = np.ascontiguousarray(event[np.newaxis, :])#1,4,h,w
            sample['image'] = torch.from_numpy(image).float()
            sample['label'] = torch.from_numpy(label).float()
            sample['event'] = torch.from_numpy(event).float()
        elif 'davis240c' in sample.keys():
            image, label, event = sample['image'], sample['label'], sample['event']
            image = np.ascontiguousarray(image[np.newaxis, :]).astype(np.float32) #1,1,h,w
            label = np.ascontiguousarray(label[np.newaxis, :]).astype(np.float32) #1,4,1,h,w
            event = np.ascontiguousarray(event[np.newaxis, :]).astype(np.float32) #1,16,h,w
            sample['image'] = torch.from_numpy(image)
            sample['label'] = torch.from_numpy(label)
            sample['event'] = torch.from_numpy(event)
        else:
            image, label, event = sample['image'], sample['label'], sample['event']
            image = np.ascontiguousarray(image.transpose((2, 0, 1))[np.newaxis, :])#1,3,h,w
            label = np.ascontiguousarray(label.transpose((3, 2, 0, 1))[np.newaxis, :])#1,4,3,h,w
            event = np.ascontiguousarray(event.transpose((2, 0, 1))[np.newaxis, :])#1,4,h,w
            sample['image'] = torch.from_numpy(image).float()
            sample['label'] = torch.from_numpy(label).float()
            sample['event'] = torch.from_numpy(event).float()
        return sample


def normalize(x, centralize=False, normalize=False, val_range=255.0, iseventfocus=False, istsi=False, is5d=False):
    if iseventfocus:
        if istsi:
            x = x/12
        else:
            x = x/127
        return x
    if normalize:
        if is5d:
            x = x/255
            return x
        else:
            b,c,h,w = x.shape
            if c == 3:
                x = x / val_range
                return x
            elif c==12:
                x = x/255
                return x
            elif c==4:
                x = x/255
                return x
            elif c==8:
                x = x/255
                return x
            elif c==10:
                x = x/255
                return x
            elif c==48:
                x = x/10
                return x
            else:
                x = x/255
                return x


def normalize_reverse(x, centralize=False, normalize=False, val_range=255.0):
    if normalize:
        x = x * val_range

    return x

def savetensor(x):
    path = join(self.save_dir, 'sample.npy')
    with open(path, 'wb') as f:
        np.save(f, x)
