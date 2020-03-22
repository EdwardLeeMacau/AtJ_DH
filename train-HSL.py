"""
  Filename       [ train-HSV.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ Default training process with tensorboardX ]
"""

import argparse
import os
import random
import time
from math import log10

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip,
                                    RandomVerticalFlip, ToTensor)

# import model.AtJ_At as atj
from cmdparser import parser
from datasets.data import MultiChannelDatasetFromFolder
from misc_train import DehazeLoss, HazeLoss
# from model.At_model import Dense
from model.My_At_model_HSL import Dense_At
from model.perceptual import Perceptual, vgg16ca
from tensorboardX import SummaryWriter
from transforms.ssim import ssim as SSIM
from utils.utils import norm_ip, norm_range

MEAN, STD = None, None

def train(data, target, model: nn.Module, optimizer: optim.Optimizer, criterion, perceptual=None, gamma=0, kappa=0):
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
    
    # ReHaze - Data Pair
    if gamma != 0: 
        amap, tmap = output[1], output[2]
        rehaze = target * tmap + amap * (1 - tmap)
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
    output = model(data)

    # DeHaze - Target Pair
    dehaze = output[0]
    loss = DehazeLoss(dehaze, target, criterion, perceptual) 
    
    # ReHaze - Data Pair
    if gamma != 0:
        amap, tmap = output[1], output[2]
        rehaze = target * tmap + amap * (1 - tmap)
        loss += gamma * HazeLoss(rehaze, data, criterion, perceptual)    

    return dehaze, loss.item()

def getDataLoaders(opt, train_transform, val_transform):
    """
    Parameters
    ----------
    opt : Namespace

    train_transform, val_transform : torchvision.transform

    Return
    ------
    ntire_train_loader, ntire_val_loader : DataLoader
        TrainLoader and ValidationLoader
    """
    ntire_train_loader = DataLoader(
        dataset=MultiChannelDatasetFromFolder(opt.dataroot, transform=train_transform), 
        num_workers=opt.workers, 
        batch_size=opt.batchSize, 
        pin_memory=True, 
        shuffle=True
    )

    ntire_val_loader = DataLoader(
        dataset=MultiChannelDatasetFromFolder(opt.valDataroot, transform=val_transform), 
        num_workers=opt.workers, 
        batch_size=opt.valBatchSize, 
        pin_memory=True, 
        shuffle=True
    )

    return ntire_train_loader, ntire_val_loader

def main():
    opt = parser.parse_args()

    if os.path.exists(opt.outdir):
        if len(os.listdir(opt.outdir)) != 0:
            raise ValueError("Directory --outdir {} exists and not empty.".format(opt.outdir))

    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    with open(os.path.join(opt.outdir, "record.txt"), 'w') as f:
        for key, value in vars(opt).items():
            print("{:20} {:>50}".format(key, str(value)))
            f.write("{:20} {:>50}\n".format(key, str(value)))

    train_transform = Compose([
        # RandomHorizontalFlip(),
        # RandomVerticalFlip(),
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

    # NOTE weight for L2 and Lp (i.e. Eq.(3) in the paper)
    gamma = opt.lambdaG
    kappa = opt.lambdaK

    epochs = opt.niter
    print_every = opt.print_every
    val_every = opt.val_every

    startepoch = 0
    running_loss = 0.0
    valLoss, valPSNR, valSSIM = 0.0, 0.0, 0.0

    min_valloss, min_valloss_epoch = 20.0, 0
    max_valpsnr, max_valpsnr_epoch = 0.0, 0
    max_valssim, max_valssim_epoch = 0.0, 0

    # Deploy model and perceptual model
    model = Dense_At()
    net_vgg = None

    if opt.netG:
        model.load_state_dict(torch.load(opt.netG)['model'])

    if kappa != 0:
        net_vgg = vgg16ca()
        net_vgg.eval()

    # Set GPU (Data parallel)
    if len(opt.gpus) > 1:
        raise NotImplementedError

        model = nn.DataParallel(model, device_ids=opt.gpus)
        net_vgg = nn.DataParallel(net_vgg, device_ids=opt.gpus)

    else:
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus[0])

        model.cuda()
        if kappa != 0: 
            net_vgg.cuda()

    MEAN = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    STD  = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    # Freezing Encoder
    # for i, child in enumerate(model.children()):
    #     if i == 12: 
    #         break
    #
    #     for param in child.parameters(): 
    #         param.requires_grad = False 

    # Setup Optimizer and Scheduler
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr = opt.learningRate, 
        weight_decay=5e-5
    )

    scheduler = StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    # Main Loop of training
    t0 = time.time()

    with SummaryWriter(comment=opt.comment) as writer:
        for epoch in range(startepoch, epochs):
            model.train() 

            for i, (data, target) in enumerate(dataloader, 1): 
                # ----------------------------------------------------- #
                # Mapping DEHAZE - GT, with Perceptual Loss             #
                # ----------------------------------------------------- #
                data, target = data.float().cuda(), target.float().cuda() 

                optimizer.zero_grad()

                outputs, A, t = model(data)

                L2 = criterionMSE(outputs, target)

                if kappa != 0:
                    outputsvgg = net_vgg(outputs)
                    targetvgg  = net_vgg(target)
                    Lp = sum([criterionMSE(outputVGG, targetVGG) for (outputVGG, targetVGG) in zip(outputsvgg, targetvgg)])
                    loss = L2 + kappa * Lp
                else:
                    loss = L2

                # ----------------------------------------------------- #
                # Mapping REHAZE - HAZE(I), with Perceptual Loss        #
                # ----------------------------------------------------- #
                if gamma != 0:
                    rehaze = target * t + A * (1 - t)
                    
                    RehazeL2 = criterionMSE(rehaze, data)

                    if kappa != 0:
                        rehazesvgg = net_vgg(rehaze)
                        datasvgg   = net_vgg(data)
                        RehazeLp   = sum([criterionMSE(rehazeVGG, dataVGG) for (rehazeVGG, dataVGG) in zip(rehazesvgg, datasvgg)])
                        loss += gamma * RehazeL2 + kappa * RehazeLp
                    else:
                        loss += gamma * RehazeL2
                    
                # Update parameters
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # ----------------------------------------------------- #
                # Print Loop                                            #
                # ----------------------------------------------------- #
                if (i % print_every == 0):
                    running_loss = running_loss / print_every

                    print('Epoch: {:2d} ({:3d} h {:3d} min {:3d} s) [{:5d} / {:5d}] loss: {:.3f}'.format(
                        epoch + 1, int((time.time() - t0) // 3600), int(((time.time() - t0) // 60) % 60), int(((time.time()) - t0) % 60),
                        i, len(dataloader), running_loss))

                    writer.add_scalar('./scalar/trainLoss', running_loss, epoch * len(dataloader) + i)

                    # Reset Value
                    running_loss = 0

                # ----------------------------------------------------- #
                # Validation Loop                                       #
                # ----------------------------------------------------- #
                if (i % val_every == 0):
                    model.eval() 

                    # Reset Value                
                    valLoss, valPSNR, valSSIM = 0.0, 0.0, 0.0

                    with torch.no_grad():
                        for j, (data, target) in enumerate(valDataloader, 1):
                            data, target = data.float().cuda(), target.float().cuda()

                            # MSE Loss Only
                            output = model(data)[0]

                            # Back to domain 0 ~ 1
                            target.mul_(torch.Tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)).add_(torch.Tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1))
                            output.mul_(torch.Tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)).add_(torch.Tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1))

                            loss = criterionMSE(output, target)

                            # PSNR / SSIM in torch.Tensor
                            psnr = 10 * log10(1 / loss)
                            ssim = SSIM(output, target)
                            
                            valLoss += loss
                            valPSNR += psnr
                            valSSIM += ssim 

                        # Print Summary
                        valLoss = valLoss / len(valDataloader)
                        valPSNR = valPSNR / len(valDataloader)
                        valSSIM = valSSIM / len(valDataloader)

                        writer.add_scalar('./scalar/valLoss', valLoss, epoch * len(dataloader) + i)
                        writer.add_scalar('./scalar/valPSNR', valPSNR, epoch * len(dataloader) + i)
                        writer.add_scalar('./scalar/valSSIM', valSSIM, epoch * len(dataloader) + i)
                       
                        # Save if update the best
                        if valLoss < min_valloss:
                            min_valloss = valLoss
                            min_valloss_epoch = epoch + 1
                        
                        if valPSNR > max_valpsnr:
                            max_valpsnr = valPSNR
                            max_valpsnr_epoch = epoch + 1
                        
                            torch.save(
                                {
                                    'model': model.state_dict(),
                                    'epoch': epoch,
                                    'optimizer': optimizer.state_dict(),
                                    'scheduler': scheduler.state_dict(),
                                }, 
                                os.path.join(opt.outdir, 'AtJ_DH_MaxCKPT.pth')
                            ) 

                        if valSSIM > max_valssim:
                            max_valssim = valSSIM
                            max_valssim_epoch = epoch + 1

                    # Show Message
                    print('>> Epoch {:d} VALIDATION: Loss: {:.3f}, PSNR: {:.3f}, SSIM: {:.3f}'.format(
                        epoch + 1, valLoss, valPSNR, valSSIM))

                    print('>> Best Epoch: {:d}, PSNR: {:.3f}'.format(
                        max_valpsnr_epoch, max_valpsnr))

                    model.train()
            
            # Save checkpoint
            torch.save(
                {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, 
                os.path.join(opt.outdir, 'AtJ_DH_CKPT.pth')
            )

            # Decay Learning Rate
            scheduler.step()

    print('FINISHED TRAINING')
    torch.save(model.state_dict(), os.path.join(opt.outdir, 'AtJ_DH.pth'))

if __name__ == '__main__':
    main()
