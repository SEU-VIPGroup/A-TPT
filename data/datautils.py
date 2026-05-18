import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

ID_to_DIRNAME={
    'I': 'imagenet/images',
    'A': 'imagenet-adversarial/imagenet-a',
    'K': 'imagenet-sketch/images',
    'R': 'imagenet-rendition/imagenet-r',
    'V': 'imagenetv2/imagenetv2-matched-frequency-format-val',
    'flower102': 'oxford_flowers',
    'dtd': 'dtd',
    'pets': 'oxford_pets',
    'cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

class ImageFolder_path(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform,
    ):
        super().__init__(
            root=root,
            transform=transform
        )
        self.imgs = self.samples
    

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, torch.tensor(target).long(), path

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    else:
        raise NotImplementedError
        
    return testset

# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1, return_base=False, return_mix=False):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        if return_base and return_mix:
            return x_processed, x_processed, x_processed, np.float32(1.0)
        if return_base:
            return x_processed, x_processed
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    pure_mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        pure_mix += w[i] * preprocess(x_aug)
    mixed = m * x_processed + (1 - m) * pure_mix
    if return_base and return_mix:
        return mixed, x_processed, pure_mix, m
    if return_base:
        return mixed, x_processed
    return mixed


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1, return_base=False):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        self.return_base = return_base
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        if not self.return_base:
            views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
            return [image] + views
        else:
            views, bases, mixes, betas = [], [], [], []
            for _ in range(self.n_views):
                v, b, mix, beta = augmix(
                    x, self.preprocess, self.aug_list, self.severity, return_base=True, return_mix=True
                )
                views.append(v)
                bases.append(b)
                mixes.append(mix)
                betas.append(float(beta))
            bases = [image] + bases  # anchor base is the center-cropped tensor
            mixes = [image] + mixes  # placeholder for view0 (unused; keep indexing aligned)
            betas = [1.0] + betas    # placeholder for view0 (unused; keep indexing aligned)
            return [image] + views, bases, mixes, betas


class Post_AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1, return_base=False):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        self.return_base = return_base
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        if not self.return_base:
            views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
            return [image] + views
        else:
            views, bases, mixes, betas = [], [], [], []
            for _ in range(self.n_views):
                v, b, mix, beta = augmix(
                    x, self.preprocess, self.aug_list, self.severity, return_base=True, return_mix=True
                )
                views.append(v)
                bases.append(b)
                mixes.append(mix)
                betas.append(float(beta))
            bases = [image] + bases
            mixes = [image] + mixes
            betas = [1.0] + betas
            return [image] + views, bases, mixes, betas
