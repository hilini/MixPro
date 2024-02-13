# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from PIL import Image

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler
from .image import ObjectImage, ObjectImage_mul


try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp

def build_loader(config):
    dsets = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dset_loaders = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dsets['source_train'], dsets['target_train'] = build_dataset(is_train=True, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")

    dsets['source_val'], dsets['target_val'] = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train_source = torch.utils.data.DistributedSampler(
        dsets['source_train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_train_target = torch.utils.data.DistributedSampler(
        dsets['target_train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices_t = np.arange(dist.get_rank(), len(dsets['target_val']), dist.get_world_size())
    sampler_val_t = SubsetRandomSampler(indices_t)

    indices_s = np.arange(dist.get_rank(), len(dsets['source_val']), dist.get_world_size())
    sampler_val_s = SubsetRandomSampler(indices_s)

    dset_loaders['source_train'] = torch.utils.data.DataLoader(
        dsets['source_train'], sampler=sampler_train_source,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    dset_loaders['target_train'] = torch.utils.data.DataLoader(
        dsets['target_train'], sampler=sampler_train_target,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    dset_loaders['source_val'] = torch.utils.data.DataLoader(
        dsets['source_val'], sampler=sampler_val_s,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    dset_loaders['target_val'] = torch.utils.data.DataLoader(
        dsets['target_val'], sampler=sampler_val_t,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dsets, dset_loaders, mixup_fn

def build_loader_parallel(config):
    dsets = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dset_loaders = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dsets['source_train'], dsets['target_train'] = build_dataset(is_train=True, config=config)
    print(f"Successfully build train dataset")

    dsets['source_val'], dsets['target_val'] = build_dataset(is_train=False, config=config)
    print(f"Successfully build val dataset")


    dset_loaders['source_train'] = torch.utils.data.DataLoader(
        dsets['source_train'], shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    dset_loaders['target_train'] = torch.utils.data.DataLoader(
        dsets['target_train'], shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    dset_loaders['source_val'] = torch.utils.data.DataLoader(
        dsets['source_val'],
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    dset_loaders['target_val'] = torch.utils.data.DataLoader(
        dsets['target_val'],
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    return dsets, dset_loaders


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'pacs':
        if is_train:
            source_root = os.path.join(config.DATA.DATA_PATH, config.DATA.SOURCE + '.txt')
            source_dataset = ObjectImage('', source_root, transform)
            target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
            target_dataset = ObjectImage_mul('', target_root, transform)
            return source_dataset, target_dataset
        else:
            source_root = os.path.join(config.DATA.DATA_PATH, config.DATA.SOURCE + '.txt')
            source_dataset = ObjectImage('', source_root, transform)
            target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
            target_dataset = ObjectImage('', target_root, transform)
            return source_dataset, target_dataset


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    mean = config.DATA.MEAN
    std = config.DATA.STD
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32)),
            transforms.RandomCrop(config.DATA.IMG_SIZE),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32)),
            transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform


