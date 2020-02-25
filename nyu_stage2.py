from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# cudnn.benchmark = True
# cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import model.AtJ_At as atj #atj_model -> atj_model2


# import model.dehaze1113 as atj

# from misc import *
# from misc2 import *
#from misc3 import *
from misc_train import *
#  from misc_split import *
#import model.vgg16ca as net

import pdb
from collections import OrderedDict
import re

import time
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
# import torchvision.models.vgg as models
from skimage.measure import compare_psnr, compare_ssim

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix_notcombined',  help='')
parser.add_argument('--dataroot', required=False,
  default='./artificial_dataset/trainData', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='./artificial_dataset/valData', help='path to val dataset')
parser.add_argument('--outdir', required=False,
  default='./pretrained-model/nyu_final_At.pth', help='path to saved model')
  
parser.add_argument('--outdir_CKPT', required=False,
  default='./pretrained-model/nyu_finalCKPT_At.pth', help='path to checkpoint')
parser.add_argument('--outdir_max', required=False,
  default='./pretrained-model/nyu_maxCKPT_At_cont.pth', help='path to max checkpoint')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=480, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=480, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lambda2', type=float, default=1, help='lambda2')
parser.add_argument('--lambdaP', type=float, default=0.2, help='lambdaP')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
opt.manualSeed = random.randint(1, 10000)
# opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)



opt.dataset='pix2pix_notcombined'
dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize, # no use for originalSize now, his usage is already done in Preprocess_train
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True, # when having a sampler, this should be set false
                       seed=opt.manualSeed)
    

valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize, # opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False,
                          seed=opt.manualSeed)

criterionBCE = nn.BCELoss()
criterionCAE = nn.L1Loss() 
criterionMSE = nn.MSELoss() # L2 loss

criterionBCE.cuda()
criterionCAE.cuda()
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
net_vgg=atj.vgg16ca()
net_vgg.cuda()

# -----------------------------------------
epochs = 120
steps = 0
running_loss = 0.0
running_valloss = 0.0
running_valpsnr = 0.0
running_valssim = 0.0
print_every = 400
# netG=atj.Dense_rain_cvprw3()
netG=atj.AtJ()
netG.cuda()
### load encoder pretrained_weight ###
# encoder_dict=atj.sharedEncoder().state_dict()
pretrained_dict=torch.load('./pretrained-model/nyu_final_at.pth')
# pretrained_dict=pretrained_dict['model_state_dict']
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
state_dict=netG.state_dict()
state_dict.update(pretrained_dict)
netG.load_state_dict(state_dict)

### freezing encoder ####

for i,child in enumerate(netG.children()):
  if i < 12: 
    for param in child.parameters(): 
       param.requires_grad = False 

optimizer = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()),  lr = 0.0001, weight_decay=0.00005)

### freezing encoder ####


# optimizer = optim.Adam(netG.parameters(), lr = 0.0003, weight_decay=0.00005)  # , betas = (opt.beta1, 0.999), weight_decay=0.00005
#optimizer = optim.SGD(netG.parameters(), lr = 0.002, weight_decay=0.00005)  # , betas = (opt.beta1, 0.999), weight_decay=0.00005
scheduler = StepLR(optimizer, step_size=35, gamma=0.7)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00002, last_epoch=-1)
min_valloss = 20.0
min_valloss_epoch = 0
max_valpsnr = 0.0
max_valpsnr_epoch = 0
max_valssim = 0.0
max_valssim_epoch = 0

#### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# print(opt.netG)
  
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


#netG.apply(weights_init)

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

gamma = 1.0
kappa = lambdaP

t0 = time.time()
for epoch in range(startepoch, epochs):
    # training
    netG.train() 
    net_vgg.eval()
    # print(print_every)
    # Decay Learning Rate
    # Print Learning Rate
    print('Epoch:', epoch,'LR:', scheduler.get_lr())

    for i, data in enumerate(dataloader, 0):
       
        input_cpu, A_cpu,t_cpu = data # data is a list
        input, input_A, input_t = input_cpu.float().cuda(),A_cpu.float().cuda() ,t_cpu.float().cuda() # hazy, gt
        # input, target = Variable(input_cpu), Variable(target_cpu)

        optimizer.zero_grad()

        outputs = netG(input)# Important~~(!KING)
        # outputs = outputs[0]
        #print(outputs.shape)
        outputsvgg_A = net_vgg(outputs[1]) # for the use of perceptual loss Lp (layer relu3_1)....weights should not be updated!!
        outputsvgg_t = net_vgg(outputs[2]) # for the use of perceptual loss Lp (layer relu3_1)....weights should not be updated!!

        targetvgg_A = net_vgg(input_A)
        targetvgg_t = net_vgg(input_t)

        #print(target.shape)
       
        L2 = criterionMSE(outputs[1], input_A) # this is an object not scalar!!
        L2 =L2+ criterionMSE(outputs[2], input_t) # this is an object not scalar!!

        # Lp = criterionMSE(outputsvgg, targetvgg) # targetvgg.detach()!!!!
        Lp=criterionMSE(outputsvgg_A[0],targetvgg_A[0])
        Lp+=criterionMSE(outputsvgg_t[0],targetvgg_t[0])

        for j in range(1,3):
            Lp+=criterionMSE(outputsvgg_A[j],targetvgg_A[j])
            Lp+=criterionMSE(outputsvgg_t[j],targetvgg_t[j])
        loss = lambda2*L2 + kappa*Lp

        
       
        # print gradient!!
        # grad_log = open("gradientLP2.log", "a")
        # grad_log.truncate(0)
        # for n, p in netG.named_parameters():
        #   if(p.requires_grad) and ("weight" in n) and (p.grad is not None):
        #       grad_log.write( n + ": " + str(p.grad.abs().mean()) )
        # grad_log.close()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        # print(i,print_every-1)
        
        if i%(print_every) == (print_every-1): # traindata_size==12000==3000iteration(batchsize==4) one epoch print 3000/500=6 times
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/print_every))
            running_loss = 0.0
           
            #### skip validation
            #print('skip validation...')
            #continue
            ################### skipped  ######################
            # validationvalidation
           
            netG.eval() 
            # net_vgg.eval()
            with torch.no_grad():
              for j, vdata in enumerate(valDataloader, 0):
                  # print("j",j)
                  val_input_cpu, val_A_cpu,val_t_cpu = vdata # data is a list
                  val_input, val_input_A, val_input_t = val_input_cpu.float().cuda(),val_A_cpu.float().cuda() ,val_t_cpu.float().cuda() # hazy, gt
                  # input, target = Variable(input_cpu), Variable(target_cpu)


                  val_outputs = netG(input)# Important~~(!KING)
                  val_outputsvgg_A = net_vgg(val_outputs[1]) # for the use of perceptual loss Lp (layer relu3_1)....weights should not be updated!!
                  val_outputsvgg_t = net_vgg(val_outputs[2]) # for the use of perceptual loss Lp (layer relu3_1)....weights should not be updated!!

                  val_targetvgg_A = net_vgg(val_input_A)
                  val_targetvgg_t = net_vgg(val_input_t)

                  #print(target.shape)
                
                  val_L2 = criterionMSE(val_outputs[1], val_input_A) # this is an object not scalar!!
                  val_L2 =val_L2+ criterionMSE(val_outputs[2], val_input_t) # this is an object not scalar!!

                  # Lp = criterionMSE(val_outputsvgg, targetvgg) # targetvgg.detach()!!!!
                  val_Lp=criterionMSE(val_outputsvgg_A[0],val_targetvgg_A[0])
                  val_Lp+=criterionMSE(val_outputsvgg_t[0],val_targetvgg_t[0])

                  for j in range(1,3):
                      val_Lp+=criterionMSE(val_outputsvgg_A[j],targetvgg_A[j])
                      val_Lp+=criterionMSE(val_outputsvgg_t[j],targetvgg_t[j])
                  val_loss = lambda2*val_L2 + kappa*val_Lp

                  running_valloss += val_loss
                  # accuracy curve(psnr/ssim)
               
                  # does psnr/ssim consume too much time??maybe, just compute three times per validation
                  if j%20 == 19 :
                 
                    running_valpsnr += psnr
                    running_valssim += ssim 
                                    

                  if j%60 == 59 : # valdata_size == 3000
                      running_valloss = running_valloss/3000
                      
                      print('[epoch %d] valloss: %.3f' % (epoch+1, running_valloss))
                 
                      if running_valloss<min_valloss:
                          min_valloss = running_valloss
                          min_valloss_epoch = epoch+1
                     
                          torch.save({'model_state_dict': netG.state_dict(),
                                      'epoch': epoch,
                                      'optimizer_state_dict': optimizer.state_dict(),
                                      'scheduler_state_dict': scheduler.state_dict(),
                                      'loss': loss
                                      }, opt.outdir_max) # save model at the largest valpsnr
                    
                      running_valloss = 0.0
                      running_valpsnr = 0.0
                      running_val_loss= 0.0
            netG.train()
            # net_vgg.train()

    print('min_valloss_epoch %d: %.3f' % (min_valloss_epoch, min_valloss))

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
