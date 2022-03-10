from robustness.datasets import DataSet, CIFAR, ImageNet
from robustness import data_augmentation as da
import torch as ch
from . import constants as cs
from torchvision.datasets import CIFAR100
from .caltech import Caltech101, Caltech256
from .cub import CUB
from . import cub_transform as transforms

from . import aircraft, food_101, dtd
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

class ImageNetTransfer(DataSet):
    def __init__(self, data_path, **kwargs):
        ds_kwargs = {
            'num_classes': kwargs['num_classes'],
            'mean': ch.tensor(kwargs['mean']),
            'custom_class': None,
            'std': ch.tensor(kwargs['std']),
            'transform_train': cs.TRAIN_TRANSFORMS,
            'label_mapping': None,
            'transform_test': cs.TEST_TRANSFORMS
        }
        super(ImageNetTransfer, self).__init__(kwargs['name'], data_path, **ds_kwargs)

class TransformedDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.transform = transform
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample, label = self.ds[idx]
        if self.transform:
            sample = self.transform(sample)
            if sample.shape[0] == 1:
                sample = sample.repeat(3,1,1)
        return sample, label


def make_loaders_imgnet_128k(batch_size, workers):
    ds = ImageNet(cs.IMGNET_128k_PATH)
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_imgnet_full(batch_size, workers):
    ds = ImageNet(cs.IMGNET_FULL_PATH)
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_cub(batch_size, workers):
    ds = ImageNetTransfer(cs.CUB_PATH, num_classes=200, name='cub',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_CIFAR100(batch_size, workers, subset):
    ds = ImageNetTransfer('/tmp', num_classes=100, name='cifar100',
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761])
    ds.custom_class = CIFAR100
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers, subset=subset)

def make_loaders_caltech256(batch_size, workers):
    ds = Caltech256(cs.CALTECH256_PATH, download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 60

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    train_set = TransformedDataset(train_set, transform=cs.TRAIN_TRANSFORMS)
    test_set = TransformedDataset(test_set, transform=cs.TEST_TRANSFORMS)

    return 257, [DataLoader(d, batch_size=batch_size, shuffle=True,
                num_workers=workers) for d in (train_set, test_set)]


def make_loaders_dtd(batch_size, workers):
        train_set = dtd.DTD(train=True)
        val_set = dtd.DTD(train=False)
        return 47, [DataLoader(ds, batch_size=batch_size, shuffle=True,
                num_workers=workers) for ds in (train_set, val_set)]

DS_TO_FUNC = {
    "dtd": make_loaders_dtd,
    "cifar100": make_loaders_CIFAR100,
    "cub": make_loaders_cub,
    "caltech256": make_loaders_caltech256,
    "imgnet_128k": make_loaders_imgnet_128k,
    "imgnet_full": make_loaders_imgnet_full
}

def make_loaders(ds, batch_size, workers, subset):
    if ds in ['cifar10', 'cifar100']:
        return DS_TO_FUNC[ds](batch_size, workers, subset)

    if subset: raise Exception(f'Subset not supported for the {ds} dataset')
    return DS_TO_FUNC[ds](batch_size, workers)
