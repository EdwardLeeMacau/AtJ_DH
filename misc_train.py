"""
  Filename       [ misc_train.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ] 
"""

import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

#import transforms.pix2pix as transforms
import transforms.pix1pix as transforms
from datasets.pix2pix_notcombined import pix2pix_notcombined as commonDataset

# valDataloader = getLoader(
#     datasetName=opt.dataset,
#     dataroot=opt.valDataroot,
#     originalSize=opt.imageSize, # opt.originalSize,
#     imageSize=opt.imageSize,
#     batchSize=opt.valBatchSize,
#     workers=opt.workers,
#     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#     split='val',
#     shuffle=False,
#     seed=opt.manualSeed
# )


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image
    
        if self.num_imgs < self.pool_size:
            self.images.append(image.clone())
            self.num_imgs += 1
            return image
        else:
            if np.random.uniform(0,1) > 0.5:
                random_id = np.random.randint(self.pool_size, size=1)[0]
                tmp = self.images[random_id].clone()
                self.images[random_id] = image.clone()
                return tmp
            else:
                return image

def create_exp_dir(exp):
    try:
        os.makedirs(exp)
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    
    return True


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_uniform(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # seems that can't use xavier nor kaiming normal
        m.bias.data.fill_(0)

def getLoader(datasetName, dataroot, originalSize, imageSize, batchSize=64, workers=4,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', shuffle=True, seed=None):
    """
    :param datasetName:

    :param dataroot:
    """

    dataset = commonDataset(
        root=dataroot,
        transform=transforms.Compose([
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

def saveTrainingCurve(train_loss, val_loss, psnr, ssim, epoch, fname=None, linewidth=1):
    """
    Plot out learning rate, training loss, validation loss and PSNR.

    Parameters
    ----------
    train_loss, val_loss, psnr, ssim : 1D-array like
        (...)

    linewidth : float
        Default linewidth
    """
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(19.2, 10.8))

    # Linear scale of loss curve
    ax = axs[0]
    line1, = ax.plot(
        np.linspace(0, epoch, len(val_loss)), 
        val_loss, 
        label="Validation Loss", 
        color='red', 
        linewidth=linewidth
    )
    line2, = ax.plot(
        np.linspace(0, epoch, len(train_loss)),
        train_loss, 
        label="Train Loss", 
        color='blue', 
        linewidth=linewidth
    )
    ax.set_xlabel("Epoch(s)")
    ax.set_ylabel("Image Loss")
    ax.set_title("Loss")

    ax.legend(handles=(line1, line2, ))

    # Linear scale of PSNR, SSIM
    ax = axs[1]
    line1, = ax.plot(
        np.linspace(0, epoch, len(psnr)),
        psnr, 
        label="PSNR", 
        color='blue', 
        linewidth=linewidth
    )
    ax.set_xlabel("Epochs(s)")
    ax.set_ylabel("PSNR")
    ax.set_title("Validation Performance")
    ax.legend()
    
    # Linear scale of SSIM
    ax = axs[1].twinx()
    line2, = ax.plot(
        np.linspace(0, epoch, len(ssim)),
        ssim, 
        label="SSIM", 
        color='red', 
        linewidth=linewidth
    )

    ax.set_ylabel("SSIM")
    ax.legend()
 
    if fname is not None:
        plt.savefig(fname)

    plt.clf()

    return



def adjust_learning_rate(optimizer, init_lr, epoch, factor, every):
    """
    :param optimizer: 

    :param init_learning_rate:
    
    :param epoch:

    :param every:
    """
    lrd = init_lr / every
    old_lr = optimizer.param_groups[0]['lr']
    # linearly decaying lr
    lr = old_lr - lrd
    if lr < 0: lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
