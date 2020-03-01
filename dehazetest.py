"""
  Filename       [ dehaze_test.py ]
  PackageName    [ AtJ_DH ]
"""

from __future__ import print_function

import argparse
import os
import pdb
import random
import re
import sys
import time
from collections import OrderedDict

from PIL import Image

import model.AtJ_At as net
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from misc import *
from torch.autograd import Variable

from cmdparser import parser

# import models.dehaze1113 as net
# import dehaze1113 as net
# import model.AtJ_At as net

# cudnn.benchmark = True
# cudnn.fastest = True

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    print("dehazetest.py start!!")

    torch.cuda.set_device(0)

    opt = parser.parse_args()
    print(opt)

    create_exp_dir(opt.exp)
    opt.manualSeed = random.randint(1, 10000)
    # opt.manualSeed = 101
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    print("Random Seed: ", opt.manualSeed)

    opt.dataset = 'pix2pix_val_temp'

    valDataloader = getLoader(
        opt.dataset,
        opt.valDataroot,
        opt.imageSize, #opt.originalSize,
        opt.imageSize,
        opt.valBatchSize,
        opt.workers,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        split='val',
        shuffle=False,
        seed=opt.manualSeed
    )

    inputChannelSize = opt.inputChannelSize
    outputChannelSize= opt.outputChannelSize

    # netG=net.Dense_rain_cvprw3()
    # netG.apply(weights_init)
    netG = net.AtJ()

    print("Before modify!!\n")
    # print("Pretrained-model:\n")


    # if opt.netG != '':
    #   pretrained_state_dict = torch.load(opt.netG)
    #   print("Start modify!!\n")
    #   new_state_dict = OrderedDict()
    #   for i, (key, val) in enumerate(pretrained_state_dict.items()):
    #     x = re.findall("dense_block[0-9]+\.denselayer[0-9]+\.(?:norm|conv)\.[0-9]+\.\S+", key) # x is list of string
    #     if x :
    #       normx = re.findall("norm\.[0-9]+",x[0]) # normx[0] == 'norm.1' or 'norm.2'
    #       convx = re.findall("conv\.[0-9]+",x[0]) # convx[0] == 'conv.1' or 'conv.2'
    #       if normx :
    #           result_key = re.sub('norm\.[0-9]+','norm'+normx[0][5],x[0])
    #       elif convx :
    #           result_key = re.sub('conv\.[0-9]+','conv'+convx[0][5],x[0])
    #       new_state_dict[result_key] = val
    #     else :
    #       new_state_dict[key] = val
    pretrained_state_dict = torch.load(opt.netG)
    if "forstage2" in opt.netG:
        netG.load_state_dict(pretrained_state_dict)
    else :
        netG.load_state_dict(pretrained_state_dict['model_state_dict'])

    #netG.load_state_dict(new_state_dict)

    netG.eval()
    # netG.train()

    criterionBCE = nn.BCELoss()
    criterionCAE = nn.L1Loss()

    val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
    val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

    # val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
    # val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

    # Switch on CUDA
    netG.cuda()
    criterionBCE.cuda()
    criterionCAE.cuda()

    # val_target, val_input = val_target.cuda(), val_input.cuda()

    t0 = time.time()
    for i, data_val in enumerate(valDataloader, 0):
        val_input_cpu, val_target_cpu, path = data_val

        val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
        val_target = Variable(val_target_cpu, volatile=True)
        
        for _ in range(val_input.size(0)):
            x_hat_val = netG(val_target)[0]

            # val_batch_output[idx,:,:,:].copy_(x_hat_val.data)
        # vutils.save_image(x_hat_val.data, './image_heavy/'+str(i)+'.jpg', normalize=True, scale_each=False,  padding=0, nrow=1)
        tensor = x_hat_val.data.cpu()

        if not os.path.exists(opt.outdir):
            os.makedirs(opt.outdir)

        filename = os.path.join(opt.outdir, str(i) + '.png')

        tensor = torch.squeeze(tensor)
        tensor = norm_range(tensor, None)
        print('Patch:'+str(i))

        ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        im = Image.fromarray(ndarr)
        im.save(filename)
        t1 = time.time()
        print('running time:'+str(t1-t0))

if __name__ == "__main__":
    main()
