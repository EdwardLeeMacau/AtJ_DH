"""
  Filename       [ misc.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ] 
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import transforms.pix2pix as transforms

def getLoader(datasetName, dataroot, originalSize, imageSize, batchSize=64, workers=4,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', shuffle=True, seed=None):

    if datasetName == 'pix2pix':
        from datasets.pix2pix import pix2pix as commonDataset
    elif datasetName == 'pix2pix_val':
        from datasets.pix2pix_val import pix2pix_val as commonDataset
    elif datasetName == 'pix2pix_temp':
        from datasets.pix2pix_temp import pix2pix_temp as commonDataset
    elif datasetName == 'pix2pix_val_temp':
        from datasets.pix2pix_val_temp import pix2pix_val_temp as commonDataset
    elif datasetName == 'pix2pix_val_full':
        from datasets.pix2pix_val_full import pix2pix_val_full as commonDataset

    if split == 'train':
        dataset = commonDataset(
            root=dataroot,
            transform=transforms.Compose([
                transforms.Scale(originalSize),
                transforms.RandomCrop(imageSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            seed=seed
        )

    elif split=='full':
        dataset = commonDataset(
            root=dataroot,
            transform=transforms.Compose([
                #transforms.Scale(originalSize),
                # transforms.CenterCrop(imageSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            seed=seed
        )

    else:
        dataset = commonDataset(
            root=dataroot,
            transform=transforms.Compose([
                transforms.Scale(originalSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            seed=seed)

    dataloader = DataLoader(
        dataset, 
        batch_size=batchSize, 
        shuffle=shuffle, 
        num_workers=int(workers)
    )
    
    return dataloader
