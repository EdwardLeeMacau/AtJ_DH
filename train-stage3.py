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
from tensorboardX import SummaryWriter

import model.AtJ_At as atj
from cmdparser import parser
from datasets.data import DatasetFromFolder
from misc_train import *
from model.At_model import Dense
from model.perceptual import vgg16ca, perceptual
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
        dehaze_p1, dehaze_p2, dehaze_p3 = perceptual(dehaze)
        target_p1, target_p2, target_p3 = perceptual(target)

        loss += kappa * (criterion(dehaze_p1, target_p1) + criterion(dehaze_p2, target_p2) + criterion(dehaze_p3, target_p3))
    
    return loss

def HazeLoss(rehaze, target, criterion, perceptual=None, kappa=0):
    """
    Parameters
    ----------
    perceptual : optional
        Perceptual loss is applied if nn.Module is provided.
    
    kappa : float
        The ratio of criterion and perceptual loss.
    """
    loss = criterion(rehaze, target)

    if perceptual is not None: 
        rehaze_p1, rehaze_p2, rehaze_p3 = perceptual(rehaze)
        target_p1, target_p2, target_p3 = perceptual(target)
        
        loss += kappa * (criterion(rehaze_p1, target_p1) + criterion(rehaze_p2, target_p2) + criterion(rehaze_p3, target_p3))

    return loss

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
    create_exp_dir(opt.exp)

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    for key, value in vars(opt).items():
        print("{:20} {:>50}".format(key, str(value)))

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

    net_vgg = None
    if opt.lambdaK != 0:
        net_vgg = vgg16ca()
        net_vgg.cuda()
        net_vgg.eval()

    epochs = opt.niter
    print_every = opt.print_every
    val_every = opt.val_every

    running_loss = 0.0
    valLoss, valPSNR, valSSIM = 0.0, 0.0, 0.0

    # netG = atj.AtJ()
    model = Dense()

    if opt.netG :
        try:
            model.load_state_dict(torch.load(opt.netG)['model'])
        except:
            raise ValueError('Fail to load netG, check {}'.format(opt.netG))

    startepoch = 0
    model.cuda()

    # freezing encoder
    for i, child in enumerate(model.children()):
        if i == 12: break

        for param in child.parameters(): 
            param.requires_grad = False 

    # Setup Optimizer and Scheduler
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

    with SummaryWriter(comment='AtJ_DH') as writer:
        for epoch in range(startepoch, epochs):
            model.train() 

            # Print Learning Rate
            print('Epoch:', epoch, 'LR:', scheduler.get_lr())

            for i, (data, target) in enumerate(dataloader, 1): 
                data, target = data.float().cuda(), target.float().cuda() 

                loss = train(data, target, model, optimizer, criterionMSE, net_vgg, gamma=gamma, kappa=kappa)

                running_loss += loss
                
                # Print Loop
                if (i % print_every == 0):
                    running_loss = running_loss / print_every

                    print('Epoch: {:2d} ({:3d} h {:3d} min {:3d} s) [{:5d} / {:5d}] loss: {:.3f}'.format(
                        epoch + 1, int((time.time() - t0) // 3600), int(((time.time() - t0) // 60) % 60), int(((time.time()) - t0) % 60),
                        i, len(dataloader), running_loss))

                    writer.add_scalar('./scalar/trainLoss', running_loss, epoch * len(dataloader) + i)

                # Validation Loop
                if (i % val_every == 0):
                    model.eval() 
                    
                    with torch.no_grad():
                        for j, (data, target) in enumerate(valDataloader, 1):
                            data, target = data.float().cuda(), target.float().cuda()

                            output, loss = val(data, target, model, criterionMSE)

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

                    # Reset Value                
                    valLoss, valPSNR, valSSIM = 0.0, 0.0, 0.0

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
