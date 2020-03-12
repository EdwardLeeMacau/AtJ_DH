"""
  Filename       [ train-stage2.py ]
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
from model.perceptual import perceptual, vgg16ca
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip
from utils.utils import norm_ip, norm_range

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def DehazeLoss(dehaze, target, criterion, perceptual=None, kappa=0):
    """
    Parameters
    ----------
    perceptual : optional
        Perceptual loss is applied if nn.Module is provided.
    
    kappa : float
        The ratio of criterion and perceptual loss.
    """
    loss = criterion(dehaze, target)
    if perceptual is not None:
        loss += kappa * perceptual(dehaze, target)
    
    return loss

def HazeLoss(haze, target, criterion, perceptual=None, kappa=0):
    """
    Parameters
    ----------
    perceptual : optional
        Perceptual loss is applied if nn.Module is provided.
    
    kappa : float
        The ratio of criterion and perceptual loss.
    """
    loss = criterion(haze, target)
    if perceptual is not None: 
        loss += kappa * perceptual(haze, target)

    return loss

def train(data, amap, tmap, target, model: nn.Module, optimizer: optim.Optimizer, criterion, perceptual=None, gamma=0, kappa=0):
    """
    Parameters
    ----------
    perceptual : optional
        Perceptual loss is applied if nn.Module is provided.
    
    gamma : float
        The ratio of DeHaze - Target Pair and ReHaze - Data Pair.

    kappa : float
        The ratio of criterion and perceptual loss.
    """
    optimizer.zero_grad()

    output = model(data)

    # DeHaze - Target Pair
    dehaze = output[0]
    loss = DehazeLoss(dehaze, target, criterion, perceptual, kappa=kappa) 
    
    if amap is not None:
        loss += criterion(output[1], amap)

    if tmap is not None:
        loss += criterion(output[2], tmap)

    # ReHaze - Data Pair
    if gamma != 0: 
        rehaze = model(data)[0]
        loss += gamma * HazeLoss(rehaze, data, criterion, perceptual, kappa=kappa)

    loss.backward()
    optimizer.step()

    return loss.item()

def val(data, target, model: nn.Module, criterion, perceptual=None, gamma=0, kappa=0):
    """
    Parameters
    ----------
    perceptual : optional
        Perceptual loss is applied if nn.Module is provided.
    
    gamma : float
        The ratio of DeHaze - Target Pair and ReHaze - Data Pair.

    kappa : float
        The ratio of criterion and perceptual loss.
    """
    # DeHaze - Target Pair
    dehaze = model(data)[0]
    loss = DehazeLoss(dehaze, target, criterion, perceptual) 
    
    # ReHaze - Data Pair
    if gamma != 0: 
        rehaze = model(data)[3]
        loss += gamma * HazeLoss(rehaze, data, criterion, perceptual)    

    return dehaze, loss.item()

def getDataLoaders(opt, train_transform, val_transform):
    ntire_train_loader = DataLoader(
        dataset=DatasetFromFolder(opt.dataroot, transform=train_transform), 
        num_workers=opt.workers, 
        batch_size=opt.batchSize, 
        pin_memory=True, 
        shuffle=True
    )

    nyu_train_loader = getLoader(
        dataroot=opt.nyuDataroot,
        transform=train_transform,
        batchSize=opt.batchSize,
        workers=opt.workers,
        shuffle=True,
        seed=opt.manualSeed
    )

    ntire_val_loader = DataLoader(
        dataset=DatasetFromFolder(opt.valDataroot, transform=val_transform), 
        num_workers=opt.workers, 
        batch_size=opt.valBatchSize, 
        pin_memory=True, 
        shuffle=True
    )

    return ntire_train_loader, nyu_train_loader, ntire_val_loader

def main():
    opt = parser.parse_args()
    create_exp_dir(opt.exp)

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    for key, value in vars(opt).items():
        print("{:20} {:>50}".format(key, str(value)))

    train_transform = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(), 
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    loader1, loader2, valLoader = getDataLoaders(opt, train_transform, val_transform)

    criterionMSE = nn.MSELoss()
    criterionMSE.cuda()

    # NOTE weight for L2 and Lp (i.e. Eq.(3) in the paper)
    gamma = opt.lambdaG
    kappa = opt.lambdaK

    # Initialize VGG-16 with batch norm as Perceptual Loss
    net_vgg = None

    if kappa != 0:
        net_vgg = perceptual(vgg16ca(), nn.MSELoss())
        net_vgg.cuda()
        net_vgg.eval()

    epochs      = opt.niter
    print_every = opt.print_every
    val_every   = opt.val_every

    running_loss = 0.0
    running_valloss, running_valpsnr, running_valssim = 0.0, 0.0, 0.0

    loss_dict = {
        'trainLoss': [],
        'valLoss': [],
        'valPSNR': [],
        'valSSIM': [],
    }

    # netG = atj.AtJ()
    model = Dense()

    if opt.netG :
        try:
            model.load_state_dict(torch.load(opt.netG)['model'])
        except:
            raise ValueError('Fail to load netG, check {}'.format(opt.netG))
    else:
        startepoch = 0

    model.cuda()

    # freezing encoder
    for i, child in enumerate(model.children()):
        if i < 12: 
            for param in child.parameters(): 
                param.requires_grad = False 

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr = 0.0001, 
        weight_decay=0.00005
    )

    scheduler = StepLR(optimizer, step_size=35, gamma=0.7)
    
    min_valloss, min_valloss_epoch = 20.0, 0
    max_valpsnr, max_valpsnr_epoch = 0.0, 0
    max_valssim, max_valssim_epoch = 0.0, 0
   
    # Main Loop of training
    t0 = time.time()
    for epoch in range(startepoch, epochs):
        model.train() 

        # Print Learning Rate
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())

        for i, (patch1, patch2) in enumerate(zip(loader1, loader2), 1): 
            # For NTIRE Dataset
            data, target = patch1
            data, target = data.float().cuda(), target.float().cuda() 
            loss = train(data, None, None, target, model, optimizer, criterionMSE, net_vgg, gamma=gamma, kappa=kappa)
            running_loss += loss
            
            # For NYU Dataset
            data, amap, tmap, target = patch2
            data, amap, tmap, target = data.float().cuda(), amap.float().cuda(), tmap.float().cuda(), target.float().cuda()
            loss = train(data, amap, tmap, target, model, optimizer, criterionMSE, net_vgg, gamma=gamma, kappa=kappa)
            running_loss += loss
            
            if (i % print_every == 0): 
                running_loss = running_loss / print_every

                print('Epoch: {:2d} ({:3d} h {:3d} min {:3d} s) [{:5d} / {:5d}] loss: {:.3f}'.format(
                    epoch + 1, int((time.time() - t0) // 3600), int(((time.time() - t0) // 60) % 60), int(((time.time()) - t0) % 60),
                    i, min(len(loader1), len(loader2)), running_loss)
                )

                loss_dict['trainLoss'].append(running_loss)
                running_loss = 0.0

            # Validation Loop
            if (i % val_every == 0):
                model.eval() 
                
                with torch.no_grad():
                    for j, (data, target) in enumerate(valLoader, 1):
                        data, target = data.float().cuda(), target.float().cuda()

                        dehaze, loss = val(data, target, model, criterionMSE, None, gamma=0, kappa=0)

                        running_valloss += loss
                        
                        # tensor to ndarr to get PSNR, SSIM
                        tensors = [dehaze.data.cpu(), target.data.cpu()]
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
                        running_valpsnr += psnr
                        running_valssim += ssim 
                    
                    # Print Summary
                    running_valloss = running_valloss / len(valLoader)
                    running_valpsnr = running_valpsnr / len(valLoader)
                    running_valssim = running_valssim / len(valLoader)

                    loss_dict['valLoss'].append(running_valloss)
                    loss_dict['valPSNR'].append(running_valpsnr)
                    loss_dict['valSSIM'].append(running_valssim)

                   
                    # Save if update the best
                    if running_valloss < min_valloss:
                        min_valloss = running_valloss
                        min_valloss_epoch = epoch + 1
                    
                    # Save if update the best
                    # save model at the largest valpsnr
                    if running_valpsnr > max_valpsnr:
                        max_valpsnr = running_valpsnr
                        max_valpsnr_epoch = epoch + 1
                    
                        torch.save(
                            {
                                'model': model.state_dict(),
                                'epoch': epoch,
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'loss': loss
                            }, 
                            os.path.join(opt.outdir, 'AtJ_DH_MaxCKPT.pth')
                        ) 

                    # Save if update the best
                    if running_valssim > max_valssim:
                        max_valssim = running_valssim
                        max_valssim_epoch = epoch + 1

                    saveTrainingCurve(
                        loss_dict['trainLoss'], 
                        loss_dict['valLoss'], 
                        loss_dict['valPSNR'], 
                        loss_dict['valSSIM'], 
                        epoch + i / min(len(loader1), len(loader2)),
                        fname='loss.png'
                    )

                    # Show Message
                    print('>> Epoch {:d} VALIDATION: Loss: {:.3f}, PSNR: {:.3f}, SSIM: {:.3f}'.format(
                        epoch+1, running_valloss, running_valpsnr, running_valssim)
                    )

                    print('>> Best Epoch: {:d}, PSNR: {:.3f}'.format(
                        max_valpsnr_epoch, max_valpsnr
                    )

                    # Reset Meter
                    running_valloss = 0.0
                    running_valpsnr = 0.0
                    running_valssim = 0.0

                model.train()
       
        # Save checkpoint
        torch.save(
            {
                'model': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': loss
            }, 
            os.path.join(opt.outdir, 'AtJ_DH_CKPT.pth')
        )

        # Decay Learning Rate
        scheduler.step()
        
    print('FINISHED TRAINING')
    torch.save(model.state_dict(), os.path.join(opt.outdir, 'AtJ_DH.pth'))

if __name__ == '__main__':
    main()
