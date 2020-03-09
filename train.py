"""
  Filename       [ train.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ]
"""

from __future__ import print_function

import argparse
import os
import random
import time

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from matplotlib import pyplot as plt
import model.AtJ_At as atj
from model.At_model import Dense
from model.perceptual import vgg16ca
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
# cudnn.benchmark = True
# cudnn.fastest = True
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize
# import torchvision.utils as vutils
from cmdparser import parser
from misc_train import *
from torch import optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

# from collections import OrderedDict

from datasets.data import DatasetFromFolder
from utils.utils import norm_ip, norm_range

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


def getDataLoaders(opt, transform):
    dataloader = DataLoader(
        dataset=DatasetFromFolder(opt.dataroot, transform=transform), 
        num_workers=opt.workers, 
        batch_size=opt.batchSize, 
        pin_memory=True, 
        shuffle=True
    )

    valdataloader = DataLoader(
        dataset=DatasetFromFolder(opt.valDataroot, transform=transform), 
        num_workers=opt.workers, 
        batch_size=opt.valBatchSize, 
        pin_memory=True, 
        shuffle=True
    )

    # dataloader = getLoader(
    #     datasetName=opt.dataset,
    #     dataroot=opt.dataroot,
    #     originalSize=opt.originalSize, # no use for originalSize now, his usage is already done in Preprocess_train
    #     imageSize=opt.imageSize,
    #     batchSize=opt.batchSize,
    #     workers=opt.workers,
    #     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    #     split='train',
    #     shuffle=True, # when having a sampler, this should be set false
    #     seed=opt.manualSeed
    # )

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

    return dataloader, valdataloader, None, None

def saveTrainingCurve(train_loss, val_loss, psnr, ssim, epoch, fname=None, linewidth=1):
    """
    Plot out learning rate, training loss, validation loss and PSNR.

    Parameters
    ----------
    train_loss, perc_iter, val_loss, psnr, ssim, lr, x: 1D-array like
        (...)

    iters_per_epoch : int
        To show the iterations in the epoch

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
    # ax.legend(handles=(line1, ))

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

    img_transform = Compose([
        ToTensor(), 
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloader, valDataloader, _, _ = getDataLoaders(opt, img_transform)

    criterionMSE = nn.MSELoss()
    criterionMSE.cuda()

    # NOTE weight for L2 and Lp (i.e. Eq.(3) in the paper)
    lambda2 = opt.lambda2
    lambdaP = opt.lambdaP

    # Initialize VGG-16 with batch norm as Perceptual Loss
    net_vgg = vgg16ca()
    net_vgg.cuda()
    net_vgg.eval()

    epochs = opt.niter
    print_every = 10
    val_every = 500

    running_loss = 0.0
    running_valloss, running_valpsnr, running_valssim = 0.0, 0.0, 0.0

    loss_dict = {
        'trainLoss': [],
        'valLoss': [],
        'valPSNR': [],
        'valSSIM': [],
    }

    # netG = atj.AtJ()7
    netG = Dense()

    if opt.netG :
        try:
            netG.load_state_dict(torch.load(opt.netG)['model'])
        except:
            raise ValueError('Fail to load netG, check {}'.format(opt.netG))
    else:
        startepoch = 0

    netG.cuda()

    # freezing encoder
    for i, child in enumerate(netG.children()):
        if i < 12: 
            for param in child.parameters(): 
                param.requires_grad = False 

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, netG.parameters()), 
        lr = 0.0001, 
        weight_decay=0.00005
    )

    ### freezing encoder ####
    scheduler = StepLR(optimizer, step_size=35, gamma=0.7)

    min_valloss, min_valloss_epoch = 20.0, 0
    max_valpsnr, max_valpsnr_epoch = 0.0, 0
    max_valssim, max_valssim_epoch = 0.0, 0
      
    gamma = 1.0
    kappa = lambdaP

    # Main Loop of training
    t0 = time.time()
    for epoch in range(startepoch, epochs):
        netG.train() 

        # Decay Learning Rate
        scheduler.step()

        # Print Learning Rate
        print('Epoch:', epoch,'LR:', scheduler.get_lr())

        for i, (input, target) in enumerate(dataloader, 1): 
            input, target = input.float().cuda(), target.float().cuda() # hazy, gt

            optimizer.zero_grad()

            # Mapping Dehaze - Dehaze, with Perceptual Loss
            outputs    = netG(input)[0]
            outputsvgg = net_vgg(outputs)
            targetvgg  = net_vgg(target)
            
            L2 = criterionMSE(outputs, target)
            Lp = criterionMSE(outputsvgg[0], targetvgg[0])
    
            for j in range(1, 3):
                Lp += criterionMSE(outputsvgg[j], targetvgg[j])
            loss = lambda2*L2 + kappa*Lp
          
            # Mapping ReHaze - Haze, with Perceptual Loss
            # outputs    = netG(target)[3]
            # outputsvgg = net_vgg(outputs)
            # targetvgg  = net_vgg(input)

            # L2 = criterionMSE(outputs, input)
            # Lp = criterionMSE(outputsvgg[0], targetvgg[0])

            # for j in range(1, 3):
            #     Lp += criterionMSE(outputsvgg[j], targetvgg[j])

            # loss += gamma * L2
            # loss += kappa * Lp
            
            # Update parameters
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Validation Loop
            if i % (print_every) == 0:
                running_loss = running_loss / print_every

                print('Epoch: {} [{:5d} / {:5d}] loss: {:.3f}'.format(
                    epoch + 1, i, len(dataloader), running_loss))

                loss_dict['trainLoss'].append(running_loss)
                running_loss = 0.0

            if i % (val_every) == 0:
                netG.eval() 
                
                with torch.no_grad():
                    for j, (input, target) in enumerate(valDataloader, 1):
                        input, target = input.float().cuda(), target.float().cuda()

                        output    = netG(input)[0]
                        outputvgg = net_vgg(output)
                        targetvgg = net_vgg(target)

                        L2 = criterionMSE(output, target)
                        Lp = criterionMSE(outputvgg[0], targetvgg[0].detach())
                        
                        for index in range(1, 3):
                            Lp += criterionMSE(outputvgg[index], targetvgg[index].detach())

                        loss = lambda2 * L2.item() + lambdaP * Lp.item()
                        running_valloss += loss
                        
                        # tensor to ndarr to get PSNR, SSIM
                        tensors = [output.data.cpu(), target.data.cpu()]
                        npimgs = [] 

                        for t in tensors: 
                            t = torch.squeeze(t)
                            t = norm_range(t, None)
                            npimg = t.mul(255).byte().numpy()
                        
                            npimg = np.transpose(npimg, (1, 2, 0)) # CHW to HWC
                            npimgs.append(npimg)
                        
                        # (npimgs[0]=DH, npimgs[1]=GT)
                        psnr = peak_signal_noise_ratio(npimgs[0], npimgs[1])
                        ssim = structural_similarity(npimgs[0], npimgs[1], multichannel = True)
                        running_valpsnr += psnr
                        running_valssim += ssim 

                    running_valloss = running_valloss / len(valDataloader)
                    running_valpsnr = running_valpsnr / len(valDataloader)
                    running_valssim = running_valssim / len(valDataloader)

                    loss_dict['valLoss'].append(running_valloss)
                    loss_dict['valPSNR'].append(running_valpsnr)
                    loss_dict['valSSIM'].append(running_valssim)

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
                                'model': netG.state_dict(),
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
                        epoch + i / len(dataloader),
                        fname='loss.png'
                    )
                    
                    running_valloss = 0.0
                    running_valpsnr = 0.0
                    running_valssim = 0.0

                netG.train()

        # Print records over all epoches
        print('min_valloss_epoch %d: valloss %.3f' % (min_valloss_epoch, min_valloss))
        print('max_valpsnr_epoch %d: valpsnr %.3f' % (max_valpsnr_epoch, max_valpsnr))
        print('max_valssim_epoch %d: valssim %.3f' % (max_valssim_epoch, max_valssim))
        
        # Save checkpoint
        torch.save(
            {
                'model': netG.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': loss
            }, 
            os.path.join(opt.outdir, 'AtJ_DH_CKPT.pth')
        )

    print('FINISHED TRAINING')
    t1 = time.time()
    print('running time:' + str(t1 - t0))
    torch.save(netG.state_dict(), os.path.join(opt.outdir, 'AtJ_DH.pth'))

if __name__ == '__main__':
    main()
