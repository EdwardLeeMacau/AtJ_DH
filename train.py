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

from skimage.measure import compare_psnr, compare_ssim

import model.AtJ_At as atj
from model.At_model import Dense
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

    dataloader = DataLoader(
        dataset=DatasetFromFolder(opt.dataroot, transform=img_transform), 
        num_workers=opt.workers, 
        batch_size=opt.batchSize, 
        pin_memory=True, 
        shuffle=True
    )

    valdataloader = DataLoader(
        dataset=DatasetFromFolder(opt.valDataroot, transform=img_transform), 
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

    criterionMSE = nn.MSELoss()
    criterionMSE.cuda()

    # target = torch.FloatTensor(opt.batchSize, opt.outputChannelSize, opt.imageSize, opt.imageSize)
    # input = torch.FloatTensor(opt.batchSize, opt.inputChannelSize, opt.imageSize, opt.imageSize)
    # val_target = torch.FloatTensor(opt.batchSize, opt.outputChannelSize, opt.imageSize, opt.imageSize) # 1*3*640*640
    # val_input = torch.FloatTensor(opt.batchSize, opt.inputChannelSize, opt.imageSize, opt.imageSize)

    # target, input = target.cuda(), input.cuda()
    # val_target, val_input = val_target.cuda(), val_input.cuda()

    # NOTE weight for L2 and Lp (i.e. Eq.(3) in the paper)
    lambda2 = opt.lambda2
    lambdaP = opt.lambdaP

    # Initialize VGG-16 with batch norm as Perceptual Loss
    net_vgg = atj.vgg16ca()
    net_vgg.cuda()
    net_vgg.eval()

    # steps = 0
    epochs = opt.niter
    running_loss = 0.0
    running_valloss, running_valpsnr, running_valssim = 0.0, 0.0, 0.0
    print_every = 10
    val_every = 400

    # netG = atj.AtJ()
    netG = Dense()

    if opt.netG :
        try:
            netG.load_state_dict(torch.load(opt.netG)['model'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # loss = checkpoint['loss']
            # startepoch = checkpoint['epoch'] + 1
        except:
            raise ValueError('Fail to load netG.checkpoint, check {}'.format(opt.netG))
    else:
        startepoch = 0

    netG.cuda()

    ### freezing encoder ####

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
    # optimizer = optim.Adam(netG.parameters(), lr = 0.0003, weight_decay=0.00005)  # , betas = (opt.beta1, 0.999), weight_decay=0.00005
    # optimizer = optim.SGD(netG.parameters(), lr = 0.002, weight_decay=0.00005)  # , betas = (opt.beta1, 0.999), weight_decay=0.00005
    scheduler = StepLR(optimizer, step_size=35, gamma=0.7)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00002, last_epoch=-1)

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

        for i, data in enumerate(dataloader, 1):
        
            input, target = data
            input, target = input.float().cuda(), target.float().cuda() # hazy, gt
            # input, target = Variable(input_cpu), Variable(target_cpu)

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
                # traindata_size == 12000 == 3000iteration (batchsize==4) 
                # one epoch print 3000/500=6 times
                print('Epoch: {} [{:5d} / {:5d}] loss: {:.3f}'.format(epoch + 1, i, len(dataloader), running_loss / print_every))
                running_loss = 0.0


            if i % (val_every) == 0:
                netG.eval() 
                
                with torch.no_grad():
                    for j, (val_input_cpu, val_target_cpu) in enumerate(valDataloader, 1):
                        val_input, val_target = val_input_cpu.float().cuda(), val_target_cpu.float().cuda()
                        # val_input, val_target = Variable(val_input_cpu.cuda(), volatile = True), Variable(val_target_cpu.cuda(), volatile = True)
                        val_output    = netG(val_input)[0]
                        val_outputvgg = net_vgg(val_output) # for the use of perceptual loss Lp (layer relu3_1)....weights should not be updated!!
                        val_targetvgg = net_vgg(val_target)

                        val_L2 = criterionMSE(val_output, val_target)
                        val_Lp = criterionMSE(val_outputvgg[0], val_targetvgg[0].detach())
                        
                        for val_index in range(1, 3):
                            val_Lp += criterionMSE(val_outputvgg[val_index], val_targetvgg[val_index].detach())

                        val_loss = lambda2 * val_L2.item() + lambdaP * val_Lp.item()
                        running_valloss += val_loss
                        
                        # accuracy curve(psnr/ssim)
                        tensors = [val_output.data.cpu(), val_target.data.cpu()]
                        npimgs = [] 

                        # tensor to ndarr
                        for t in tensors: 
                            t = torch.squeeze(t)
                            t = norm_range(t, None)
                            npimg = t.mul(255).byte().numpy()
                        
                            npimg = np.transpose(npimg, (1, 2, 0)) # CHW to HWC
                            npimgs.append(npimg)
                        
                        if (j % 20 == 0):
                            psnr = compare_psnr(npimgs[0], npimgs[1]) # (npimgs[0]=DH, npimgs[1]=GT)
                            ssim = compare_ssim(npimgs[0], npimgs[1], multichannel = True)
                            running_valpsnr += psnr
                            running_valssim += ssim 
                                            
                        if (j % 60 == 0):
                            running_valloss = running_valloss / 3000
                            running_valpsnr = running_valpsnr / 3
                            running_valssim = running_valssim / 3

                            print('[epoch %d] valloss: %.3f' % (epoch+1, running_valloss))
                            print('[epoch %d] valpsnr: %.3f' % (epoch+1, running_valpsnr))
                            print('[epoch %d] valssim: %.3f' % (epoch+1, running_valssim))
                            
                            # Save if update the best
                            if running_valloss < min_valloss:
                                min_valloss = running_valloss
                                min_valloss_epoch = epoch+1
                            
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
                                    opt.outdir_max
                                ) 

                            # Save if update the best
                            if running_valssim > max_valssim:
                                max_valssim = running_valssim
                                max_valssim_epoch = epoch+1

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
            opt.outdir_CKPT
        )

    print('FINISHED TRAINING')
    t1 = time.time()
    print('running time:'+str(t1 - t0))
    torch.save(netG.state_dict(), opt.outdir)

if __name__ == '__main__':
    main()
