"""
  Filename       [ train2.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ]
"""

import argparse
import os
import random
import time

from skimage.measure import compare_psnr, compare_ssim

import model.AtJ_At as atj
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
# cudnn.benchmark = True
# cudnn.fastest = True
import torch.optim as optim
# import torchvision.utils as vutils
from cmdparser import parser
from datasets.data import DatasetFromFolder
from misc_train import *
from model.At_model import Dense
from model.perceptual import perceptual, vgg16ca
from torch import optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose, Normalize, ToTensor
from utils.utils import norm_ip, norm_range

# from collections import OrderedDict


os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def DehazeLoss(dehaze, target, criterion, perceptual=None, kappa=0):
    loss = criterion(dehaze, target)
    if perceptual is not None: loss += kappa * perceptual(dehaze, target)
    
    return loss

def HazeLoss(haze, target, criterion, perceptual=None, kappa=0):
    loss = criterion(haze, target)
    if perceptual is not None: loss += kappa * perceptual(haze, target)

    return loss

def train(data, target, model, optimizer: optim.Optimizer, criterion, perceptual=None, gamma=1.0, kappa=0.2):
    optimizer.zero_grad()

    # DeHaze - Target Pair
    dehaze = model(data)[0]
    loss = DehazeLoss(dehaze, target, criterion, perceptual) 
    
    # ReHaze - Data Pair
    if gamma != 0: 
        rehaze = model(data)[3]
        loss += gamma * HazeLoss(rehaze, data, criterion, perceptual)    

    loss.backward()
    optimizer.step()

    return loss.item()

def val(data, target, model, criterion, perceptual=None, gamma=1.0, kappa=0.2):
    # DeHaze - Target Pair
    dehaze = model(data)[0]
    loss = DehazeLoss(dehaze, target, criterion, perceptual) 
    
    # ReHaze - Data Pair
    if gamma != 0: 
        rehaze = model(data)[3]
        loss += gamma * HazeLoss(rehaze, data, criterion, perceptual)    

    return dehaze, loss.item()


def main():
    opt = parser.parse_args()
    create_exp_dir(opt.exp)

    opt.manualSeed = random.randint(1, 10000)
    # opt.manualSeed = 101
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    for key, value in vars(opt).items():
        print("{:20} {:>50}".format(key, str(value)))

    img_transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_loader = DataLoader(
        dataset=DatasetFromFolder(opt.dataroot, transform=img_transform), 
        num_workers=opt.workers, 
        batch_size=opt.batchSize, 
        pin_memory=True, 
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=DatasetFromFolder(opt.valDataroot, transform=img_transform), 
        num_workers=opt.workers, 
        batch_size=opt.valBatchSize, 
        pin_memory=True, 
        shuffle=True
    )

    criterionMSE = nn.MSELoss()
    criterionMSE.cuda()

    lambda2 = opt.lambda2
    lambdaP = opt.lambdaP
    gamma = opt.lambdaG
    kappa = opt.lambdaK

    # Initialize VGG-16 with batch norm as Perceptual Loss
    net_vgg = vgg16ca()
    net_vgg.cuda()
    net_vgg.eval()

    # Initialize VGG-16 with batch norm as Perceptual Loss
    # net_perceptual = perceptual(vgg16ca(), nn.MSELoss())
    # net_perceptual.cuda()
    # net_perceptual.eval()

    # steps = 0
    epochs = opt.niter
    running_loss = 0.0
    running_valloss, running_valpsnr, running_valssim = 0.0, 0.0, 0.0
    print_every = 10
    val_every = 400

    # netG = atj.AtJ()
    model = Dense()

    if opt.netG :
        try:
            model.load_state_dict(torch.load(opt.netG)['model'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # loss = checkpoint['loss']
            # startepoch = checkpoint['epoch'] + 1
        except:
            raise ValueError('Fail to load netG.checkpoint, check {}'.format(opt.netG))
    else:
        startepoch = 0

    model.cuda()

    ### freezing encoder ####

    for i, child in enumerate(model.children()):
        if i < 12: 
            for param in child.parameters(): 
                param.requires_grad = False 

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr = 0.0001, 
        weight_decay=0.00005
    )

    scheduler = StepLR(
        optimizer, 
        step_size=35, 
        gamma=0.7
    )
    
    min_valloss, min_valloss_epoch = 20.0, 0
    max_valpsnr, max_valpsnr_epoch = 0.0, 0
    max_valssim, max_valssim_epoch = 0.0, 0
    

    # Main Loop of training
    t0 = time.time()
    for epoch in range(startepoch, epochs):
        model.train() 

        # Decay Learning Rate
        scheduler.step()

        # Print Learning Rate
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())

        for i, (data, target) in enumerate(train_loader, 1):        
            data, target = data.float().cuda(), target.float().cuda() # hazy, gt

            loss = train(data, target, model, optimizer, criterionMSE, None, gamma=gamma, kappa=kappa)

            running_loss += loss
            
            if i % (print_every) == 0: 
                # traindata_size == 12000 == 3000iteration (batchsize==4) 
                # one epoch print 3000/500=6 times
                print('Epoch: {} [{:5d} / {:5d}] loss: {:.3f}'.format(epoch + 1, i, len(train_loader), running_loss / print_every))
                running_loss = 0.0

            # Validation Loop
            if i % (val_every) == 0:
                model.eval() 
                
                with torch.no_grad():
                    for j, (data, target) in enumerate(val_loader, 1):
                        data, target = data.float().cuda(), target.float().cuda()

                        dehaze, loss = val(data, target, model, criterionMSE, None, gamma=0, kappa=kappa)

                        running_valloss += loss
                        
                        # accuracy curve(psnr/ssim)
                        tensors = [dehaze.data.cpu(), target.data.cpu()]
                        npimgs = [] 

                        # tensor to ndarr
                        for t in tensors: 
                            t = torch.squeeze(t)
                            t = norm_range(t, None)
                            npimg = t.mul(255).byte().numpy()
                        
                            npimg = np.transpose(npimg, (1, 2, 0)) # CHW to HWC
                            npimgs.append(npimg)
                        
                        # Calculate PSNR and SSIM
                        psnr = compare_psnr(npimgs[0], npimgs[1]) # (npimgs[0]=DH, npimgs[1]=GT)
                        ssim = compare_ssim(npimgs[0], npimgs[1], multichannel = True)
                        running_valpsnr += psnr
                        running_valssim += ssim 
                    
                    # Print Summary
                    running_valloss = running_valloss / 3000
                    running_valpsnr = running_valpsnr / 3
                    running_valssim = running_valssim / 3

                    print('[epoch %d] valloss: %.3f' % (epoch+1, running_valloss))
                    print('[epoch %d] valpsnr: %.3f' % (epoch+1, running_valpsnr))
                    print('[epoch %d] valssim: %.3f' % (epoch+1, running_valssim))
                    
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
                            opt.outdir_max
                        ) 

                    # Save if update the best
                    if running_valssim > max_valssim:
                        max_valssim = running_valssim
                        max_valssim_epoch = epoch+1

                    running_valloss = 0.0
                    running_valpsnr = 0.0
                    running_valssim = 0.0

                model.train()

        # Print records over all epoches
        print('min_valloss_epoch %d: valloss %.3f' % (min_valloss_epoch, min_valloss))
        print('max_valpsnr_epoch %d: valpsnr %.3f' % (max_valpsnr_epoch, max_valpsnr))
        print('max_valssim_epoch %d: valssim %.3f' % (max_valssim_epoch, max_valssim))
        
        # Save checkpoint
        torch.save(
            {
                'model': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': loss
            }, 
            opt.outdir_CKPT
        )

    print('FINISHED TRAINING')
    t1 = time.time()
    print('running time:'+str(t1 - t0))
    torch.save(model.state_dict(), opt.outdir)

if __name__ == '__main__':
    main()
