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

import model.AtJ_J as atj_forvgg #atj_model -> atj_model2
import model.dehaze1113 as atj
#from misc import *
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
  default='./sample/trainData', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='./sample/valData', help='path to val dataset')
parser.add_argument('--outdir', required=False,
  default='./pretrained-model/nyu_J_final.pth', help='path to saved model')
parser.add_argument('--outdir_CKPT', required=False,
  default='./pretrained-model/nyu_J_finalCKPT.pth', help='path to checkpoint')
parser.add_argument('--outdir_max', required=False,
  default='./pretrained-model/nyu_J_maxCKPT.pth', help='path to max checkpoint')
parser.add_argument('--batchSize', type=int, default=7, help='input batch size')
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
parser.add_argument('--lambdaP', type=float, default=0, help='lambdaP')
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
net_vgg=atj_forvgg.vgg16ca()
net_vgg.cuda()

# -----------------------------------------
epochs = 80
steps = 0
running_loss = 0.0
running_valloss = 0.0
running_valpsnr = 0.0
running_valssim = 0.0
print_every = 400
#print_every=3
# val_print_every=3
netG=atj.Dense_rain_cvprw3()
#netG=atj.AtJ()
netG.cuda()

optimizer = optim.Adam(netG.parameters(), lr = 0.0003, weight_decay=0.00005)  # , betas = (opt.beta1, 0.999), weight_decay=0.00005
#optimizer = optim.SGD(netG.parameters(), lr = 0.002, weight_decay=0.00005)  # , betas = (opt.beta1, 0.999), weight_decay=0.00005
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
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


t0 = time.time()
for epoch in range(startepoch, epochs):
    # training
    netG.train() 
    net_vgg.eval()
    print(print_every)
    # Decay Learning Rate
    scheduler.step()
    # Print Learning Rate
    print('Epoch:', epoch,'LR:', scheduler.get_lr())

    for i, data in enumerate(dataloader, 0):
       
        input_cpu, target_cpu = data # data is a list
        input, target = input_cpu.float().cuda(), target_cpu.float().cuda()
        # input, target = Variable(input_cpu), Variable(target_cpu)

        optimizer.zero_grad()
        outputs = netG(input)# Important~~(!KING)
        
        # print(outputs)
        
        outputsvgg = net_vgg(outputs) # for the use of perceptual loss Lp (layer relu3_1)....weights should not be updated!!
        targetvgg = net_vgg(target)

        L2 = criterionMSE(outputs, target) # this is an object not scalar!!
        # Lp = criterionMSE(outputsvgg, targetvgg) # targetvgg.detach()!!!!
        Lp=criterionMSE(outputsvgg[0],targetvgg[0])
        for j in range(1,3):
            Lp+=criterionMSE(outputsvgg[j],targetvgg[j])
        loss = lambda2*L2
        loss = loss + lambdaP*Lp
        loss.backward()
        
        # print gradient!!
        # grad_log = open("gradientLP2.log", "a")
        # grad_log.truncate(0)
        # for n, p in netG.named_parameters():
        #   if(p.requires_grad) and ("weight" in n) and (p.grad is not None):
        #       grad_log.write( n + ": " + str(p.grad.abs().mean()) )
        # grad_log.close()

        optimizer.step()

        running_loss += loss.item()
        # print(i,print_every-1)
        
        if i%(print_every) == (print_every-1): # traindata_size==12000==3000iteration(batchsize==4) one epoch print 3000/500=6 times
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/print_every))
            running_loss = 0.0
            # validation
           
            netG.eval() 
            # net_vgg.eval()
            with torch.no_grad():
              for j, vdata in enumerate(valDataloader, 0):
                  # print("j",j)
                  val_input_cpu, val_target_cpu = vdata
                  val_input, val_target = val_input_cpu.float().cuda(), val_target_cpu.float().cuda()
                  # val_input, val_target = Variable(val_input_cpu.cuda(), volatile = True), Variable(val_target_cpu.cuda(), volatile = True)
                  val_output = netG(val_input)
                  val_outputvgg = net_vgg(val_output) # for the use of perceptual loss Lp (layer relu3_1)....weights should not be updated!!
                  val_targetvgg = net_vgg(val_target)
                  val_L2 = criterionMSE(val_output, val_target)
                  val_Lp = criterionMSE(val_outputvgg[0], val_targetvgg[0].detach())
                  for val_index in range(1,3):
                      val_Lp+=criterionMSE(val_outputvgg[val_index], val_targetvgg[val_index].detach())
                  val_loss = lambda2 * val_L2.item() + lambdaP * val_Lp.item()
                  running_valloss += val_loss
                  # accuracy curve(psnr/ssim)
                  tensors = [val_output.data.cpu(), val_target.data.cpu()]
                  npimgs = [] 
                  for t in tensors: # tensor to ndarr
                      t = torch.squeeze(t)
                      npimg = t.mul(255).byte().numpy()
                      # from IPython import embed
                      # embed()
                      npimg = np.transpose(npimg, (1, 2, 0)) # CHW to HWC
                      npimgs.append(npimg)
                  # does psnr/ssim consume too much time??maybe, just compute three times per validation
                  if j%20 == 19 :
                    psnr = compare_psnr(npimgs[0],npimgs[1]) # (npimgs[0]=DH, npimgs[1]=GT)
                    ssim = compare_ssim(npimgs[0],npimgs[1],multichannel = True)
                    running_valpsnr += psnr
                    running_valssim += ssim 
                                    

                  if j%60 == 59 : # valdata_size == 3000
                      running_valloss = running_valloss/3000
                      running_valpsnr = running_valpsnr/3
                      running_valssim = running_valssim/3
                      print('[epoch %d] valloss: %.3f' % (epoch+1, running_valloss))
                      print('[epoch %d] valpsnr: %.3f' % (epoch+1, running_valpsnr))
                      print('[epoch %d] valssim: %.3f' % (epoch+1, running_valssim))
                      if running_valloss<min_valloss:
                          min_valloss = running_valloss
                          min_valloss_epoch = epoch+1
                      if running_valpsnr>max_valpsnr:
                          max_valpsnr = running_valpsnr
                          max_valpsnr_epoch = epoch+1
                          torch.save({'model_state_dict': netG.state_dict(),
                                      'epoch': epoch,
                                      'optimizer_state_dict': optimizer.state_dict(),
                                      'scheduler_state_dict': scheduler.state_dict(),
                                      'loss': loss
                                      }, opt.outdir_max) # save model at the largest valpsnr
                      if running_valssim>max_valssim:
                          max_valssim = running_valssim
                          max_valssim_epoch = epoch+1
                      running_valloss = 0.0
                      running_valpsnr = 0.0
                      running_valssim = 0.0
            netG.train()
            # net_vgg.train()

    print('min_valloss_epoch %d: %.3f' % (min_valloss_epoch, min_valloss))
    print('max_valpsnr_epoch %d: %.3f' % (max_valpsnr_epoch, max_valpsnr))
    print('max_valssim_epoch %d: %.3f' % (max_valssim_epoch, max_valssim))
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
