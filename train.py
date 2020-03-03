"""
  Filename       [ stage2.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ]
"""

from __future__ import print_function

import argparse
import os
import random
import time

from skimage.measure import compare_psnr, compare_ssim

import model.AtJ_At as atj  # atj_model -> atj_model2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
# cudnn.benchmark = True
# cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from cmdparser import parser
from misc_train import *
from torch import optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

# from collections import OrderedDict



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
opt = parser.parse_args()
print(opt)

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img

def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
    return norm_ip(t, t.min(), t.max())

def main():
    create_exp_dir(opt.exp)

    opt.manualSeed = random.randint(1, 10000)
    # opt.manualSeed = 101
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    print("Random Seed: ", opt.manualSeed)

    opt.dataset = 'pix2pix_notcombined'
    dataloader = getLoader(
        datasetName=opt.dataset,
        dataroot=opt.dataroot,
        originalSize=opt.originalSize, # no use for originalSize now, his usage is already done in Preprocess_train
        imageSize=opt.imageSize,
        batchSize=opt.batchSize,
        workers=opt.workers,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        split='train',
        shuffle=True, # when having a sampler, this should be set false
        seed=opt.manualSeed
    )

    valDataloader = getLoader(
        datasetName=opt.dataset,
        dataroot=opt.valDataroot,
        originalSize=opt.imageSize, # opt.originalSize,
        imageSize=opt.imageSize,
        batchSize=opt.valBatchSize,
        workers=opt.workers,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        split='val',
        shuffle=False,
        seed=opt.manualSeed
    )

    # criterionBCE = nn.BCELoss()
    # criterionCAE = nn.L1Loss() 
    criterionMSE = nn.MSELoss()

    # criterionBCE.cuda()
    # criterionCAE.cuda()
    criterionMSE.cuda()

    target = torch.FloatTensor(opt.batchSize, opt.outputChannelSize, opt.imageSize, opt.imageSize)
    input = torch.FloatTensor(opt.batchSize, opt.inputChannelSize, opt.imageSize, opt.imageSize)
    # val_target = torch.FloatTensor(opt.batchSize, opt.outputChannelSize, opt.imageSize, opt.imageSize) # 1*3*640*640
    # val_input = torch.FloatTensor(opt.batchSize, opt.inputChannelSize, opt.imageSize, opt.imageSize)

    target, input = target.cuda(), input.cuda()
    # val_target, val_input = val_target.cuda(), val_input.cuda()

    # target = Variable(target)
    # input = Variable(input) # !!!!!!!!!OMGGGGG..................volatile cannot be true for training..........
    # val_target = Variable(val_target)
    # val_input = Variable(val_input)

    #val_target = Variable(val_target, volatile=True)
    #val_input = Variable(val_input,volatile=True)

    # NOTE weight for L2 and Lp (i.e. Eq.(3) in the paper)
    lambda2 = opt.lambda2
    lambdaP = opt.lambdaP

    # Initialize VGG-16 with batch norm
    net_vgg = atj.vgg16ca()
    net_vgg.cuda()

    # -----------------------------------------
    epochs = 120
    # steps = 0
    running_loss = 0.0
    running_valloss = 0.0
    running_valpsnr = 0.0
    running_valssim = 0.0
    print_every = 400
    # netG=atj.Dense_rain_cvprw3()
    netG = atj.AtJ()
    netG.cuda()
    ### load encoder pretrained_weight ###
    # encoder_dict=atj.sharedEncoder().state_dict()
    pretrained_dict = torch.load('./pretrained-model/forstage2.pth')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
    state_dict = netG.state_dict()
    state_dict.update(pretrained_dict)
    netG.load_state_dict(state_dict)

    ### freezing encoder ####

    for i, child in enumerate(netG.children()):
        if i < 12: 
            for param in child.parameters(): 
                param.requires_grad = False 

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr = 0.0001, weight_decay=0.00005)

    ### freezing encoder ####

    # optimizer = optim.Adam(netG.parameters(), lr = 0.0003, weight_decay=0.00005)  # , betas = (opt.beta1, 0.999), weight_decay=0.00005
    # optimizer = optim.SGD(netG.parameters(), lr = 0.002, weight_decay=0.00005)  # , betas = (opt.beta1, 0.999), weight_decay=0.00005
    scheduler = StepLR(optimizer, step_size=35, gamma=0.7)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00002, last_epoch=-1)
    min_valloss = 20.0
    min_valloss_epoch = 0
    max_valpsnr = 0.0
    max_valpsnr_epoch = 0
    max_valssim = 0.0
    max_valssim_epoch = 0
    
    if opt.netG :
        try:
            checkpoint = torch.load(opt.netG)
            netG.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            loss = checkpoint['loss']
            startepoch = checkpoint['epoch'] + 1
        except:
            print('Fail to load checkpoint, check if .pth include optimizer, scheduler...')
        else:
            print('Checkpoint loaded!')
    else:
        startepoch = 0
  
    gamma = 1.0
    kappa = lambdaP

    # Main Loop of training

    t0 = time.time()
    for epoch in range(startepoch, epochs):
        # training
        netG.train() 
        net_vgg.eval()

        # Decay Learning Rate
        scheduler.step()
        # Print Learning Rate
        print('Epoch:', epoch,'LR:', scheduler.get_lr())

        for i, data in enumerate(dataloader, 1):
        
            input, target = data
            input, target = input.float().cuda(), target.float().cuda() # hazy, gt
            # input, target = Variable(input_cpu), Variable(target_cpu)

            optimizer.zero_grad()

            outputs    = netG(input)[0]# Important~~(!KING)
            outputsvgg = net_vgg(outputs) # for the use of perceptual loss Lp (layer relu3_1)....weights should not be updated!!
            targetvgg  = net_vgg(target)
            
            L2 = criterionMSE(outputs, target) # this is an object not scalar!!
            # Lp = criterionMSE(outputsvgg, targetvgg) # targetvgg.detach()!!!!
            Lp = criterionMSE(outputsvgg[0],targetvgg[0])
    
            for j in range(1,3):
                Lp += criterionMSE(outputsvgg[j], targetvgg[j])
            loss = lambda2*L2 + kappa*Lp

          
            outputs = netG(target)
            outputs = outputs[3]
            outputsvgg = net_vgg(outputs)
            targetvgg = net_vgg(input)
            L2 = criterionMSE(outputs, input)
            Lp = criterionMSE(outputsvgg[0],targetvgg[0])
            for j in range(1,3):
                Lp += criterionMSE(outputsvgg[j],targetvgg[j])
            loss += gamma * L2
            loss += kappa * Lp
            
            loss.backward()

            optimizer.step()
    
            running_loss += loss.item()
            # print(i,print_every-1)

            # Validation Loop
            if i % (print_every) == 0: # traindata_size==12000==3000iteration(batchsize==4) one epoch print 3000/500=6 times
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/print_every))
                running_loss = 0.0
            
                netG.eval() 
                # net_vgg.eval()
                with torch.no_grad():
                    for j, vdata in enumerate(valDataloader, 1):
                        # print("j",j)
                        val_input_cpu, val_target_cpu = vdata
                        #val_input, val_target = val_input_cpu.float().cuda(), val_target_cpu.float().cuda()
                        val_input, val_target = Variable(val_input_cpu.cuda(), volatile = True), Variable(val_target_cpu.cuda(), volatile = True)
                        val_output = netG(val_input)[0]
                        val_outputvgg = net_vgg(val_output) # for the use of perceptual loss Lp (layer relu3_1)....weights should not be updated!!
                        val_targetvgg = net_vgg(val_target)
                        val_L2 = criterionMSE(val_output, val_target)
                        val_Lp = criterionMSE(val_outputvgg[0], val_targetvgg[0].detach())
                        for val_index in range(1,3):
                            val_Lp += criterionMSE(val_outputvgg[val_index], val_targetvgg[val_index].detach())
                        val_loss = lambda2 * val_L2.item() + lambdaP * val_Lp.item()
                        running_valloss += val_loss
                        # accuracy curve(psnr/ssim)
                        tensors = [val_output.data.cpu(), val_target.data.cpu()]
                        npimgs = [] 

                        for t in tensors: # tensor to ndarr
                            t = torch.squeeze(t)
                            t = norm_range(t, None)
                            npimg = t.mul(255).byte().numpy()
                        
                            npimg = np.transpose(npimg, (1, 2, 0)) # CHW to HWC
                            npimgs.append(npimg)
                        # does psnr/ssim consume too much time??maybe, just compute three times per validation
                    
                        if (j % 20 == 0):
                            psnr = compare_psnr(npimgs[0],npimgs[1]) # (npimgs[0]=DH, npimgs[1]=GT)
                            ssim = compare_ssim(npimgs[0],npimgs[1],multichannel = True)
                            running_valpsnr += psnr
                            running_valssim += ssim 
                                            
                        if (j % 60 == 0):
                            running_valloss = running_valloss/3000
                            running_valpsnr = running_valpsnr/3
                            running_valssim = running_valssim/3
                            print('[epoch %d] valloss: %.3f' % (epoch+1, running_valloss))
                            print('[epoch %d] valpsnr: %.3f' % (epoch+1, running_valpsnr))
                            print('[epoch %d] valssim: %.3f' % (epoch+1, running_valssim))
                            # Save if update the best
                            if running_valloss < min_valloss:
                                min_valloss = running_valloss
                                min_valloss_epoch = epoch+1
                            # Save if update the best
                            if running_valpsnr > max_valpsnr:
                                max_valpsnr = running_valpsnr
                                max_valpsnr_epoch = epoch + 1
                                torch.save({'model_state_dict': netG.state_dict(),
                                            'epoch': epoch,
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'scheduler_state_dict': scheduler.state_dict(),
                                            'loss': loss
                                            }, opt.outdir_max) # save model at the largest valpsnr
                            # Save if update the best
                            if running_valssim > max_valssim:
                                max_valssim = running_valssim
                                max_valssim_epoch = epoch+1
                            running_valloss = 0.0
                            running_valpsnr = 0.0
                            running_valssim = 0.0

                netG.train()

        # Print records over all epoches
        print('min_valloss_epoch %d: %.3f' % (min_valloss_epoch, min_valloss))
        print('max_valpsnr_epoch %d: %.3f' % (max_valpsnr_epoch, max_valpsnr))
        print('max_valssim_epoch %d: %.3f' % (max_valssim_epoch, max_valssim))
        
        # Save checkpoint
        torch.save({'model_state_dict': netG.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss
                    }, opt.outdir_CKPT)

    print('FINISHED TRAINING')
    t1 = time.time()
    print('running time:'+str(t1-t0))
    torch.save(netG.state_dict(), opt.outdir)

if __name__ == '__main__':
    main()
