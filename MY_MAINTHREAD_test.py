"""
  Filename       [ train-stage1.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ]
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
# cudnn.benchmark = True
# cudnn.fastest = True
import torch.optim as optim
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import model.AtJ_At as atj
from cmdparser import parser
from datasets.data import DatasetFromFolder
from misc_train import *
from model.At_model import Dense
from model.perceptual import vgg16ca
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip
from utils.utils import norm_ip, norm_range

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def getDataLoaders(opt, train_transform, val_transform):
    """
    Parameters
    ----------
    opt : Namespace

    train_transform, val_transform : torchvision.transform
    
    Return
    ------
    ntire_train_loader, ntire_val_loader : DataLoader
        
    """
    ntire_train_loader = DataLoader(
        dataset=DatasetFromFolder(opt.dataroot, transform=train_transform), 
        num_workers=opt.workers, 
        batch_size=opt.batchSize, 
        pin_memory=True, 
        shuffle=True
    )

    ntire_val_loader = DataLoader(
        dataset=DatasetFromFolder(opt.valDataroot, transform=val_transform), 
        num_workers=opt.workers, 
        batch_size=opt.valBatchSize, 
        pin_memory=True, 
        shuffle=True
    )

    return ntire_train_loader, ntire_val_loader

def main():
    opt = parser.parse_args()

    if not os.path.exists(opt.netG):
        raise ValueError("netG {} doesn't exist".format(opt.netG))

    create_exp_dir(opt.exp)

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    for key, value in vars(opt).items():
        print("{:20} {:>50}".format(key, str(value)))

    train_transform = Compose([
        ToTensor(), 
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloader, valDataloader = getDataLoaders(opt, train_transform, val_transform)

    criterionMSE = nn.MSELoss()
    criterionMSE.cuda()

    # netG = atj.AtJ()
    model = Dense()
    model.load_state_dict(torch.load(opt.netG)['model'])
    model.eval()
    model.cuda()

    # Main Loop of training
    t0 = time.time()

    with torch.no_grad():
        for j, (data, target) in enumerate(valDataloader, 1):
            data, target = data.float().cuda(), target.float().cuda()

            output = model(data)[0]
            L2 = criterionMSE(output, target)
            loss = L2.item()
            
            # tensor to ndarr to get PSNR, SSIM
            tensors = [output.data.cpu(), target.data.cpu()]
            npimgs = [] 

            for t in tensors: 
                t = torch.squeeze(t)
                t = norm_range(t, None)

                npimg = t.mul(255).byte().numpy()
                npimg = np.transpose(npimg, (1, 2, 0)) # CHW to HWC
                npimgs.append(npimg)

            # Calculate PSNR and SSIM
            psnr = peak_signal_noise_ratio(npimgs[0], npimgs[1])
            ssim = structural_similarity(npimgs[0], npimgs[1], multichannel = True)

            running_valloss += loss
            running_valpsnr += psnr
            running_valssim += ssim 

        # Print Summary
        running_valloss = running_valloss / len(valDataloader)
        running_valpsnr = running_valpsnr / len(valDataloader)
        running_valssim = running_valssim / len(valDataloader)

        
    # Show Message
    print('>> VALIDATION: Loss: {:.3f}, PSNR: {:.3f}, SSIM: {:.3f}'.format(
        running_valloss, running_valpsnr, running_valssim))
    
if __name__ == '__main__':
    main()
