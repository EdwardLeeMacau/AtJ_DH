"""
  Filename       [ dehaze_test.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ]
"""

import argparse
import os
import pdb
import random
import re
import sys
import time
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision
# import torchvision.utils as vutils
from PIL import Image
from torch.autograd import Variable

# TODO: Remember change as AtJ_At when train my own model
# import model.AtJ_At as net
import At_model as net
from cmdparser import parser
from misc import *

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
    # torch.cuda.set_device(0)

    # CmdParser
    opt = parser.parse_args()

    create_exp_dir(opt.exp)

    opt.manualSeed = random.randint(1, 10000)
    # opt.manualSeed = 101
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    print("Random Seed: ", opt.manualSeed)

    opt.dataset = 'pix2pix_val_temp'

    for key, value in vars(opt).items():
        print("{:20} {:>50}".format(key, str(value)))

    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)

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

    # netG = net.AtJ()
    netG = net.Dense()
    netG.load_state_dict(torch.load(opt.netG))

    # pretrained_state_dict = torch.load(opt.netG)
    # print(pretrained_state_dict['model'])

    # if "forstage2" in opt.netG:
    #     netG.load_state_dict(pretrained_state_dict)
    # else:
    #     netG.load_state_dict(pretrained_state_dict['model'])

    netG.eval()

    criterionBCE = nn.BCELoss()
    criterionCAE = nn.L1Loss()

    val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
    val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

    # Switch on CUDA
    netG.cuda()
    criterionBCE.cuda()
    criterionCAE.cuda()

    # val_target, val_input = val_target.cuda(), val_input.cuda()

    t0 = time.time()

    with torch.no_grad():
        for i, data_val in enumerate(valDataloader, 0):
            val_input_cpu, val_target_cpu, path = data_val

            val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
            val_target = val_target_cpu.cuda()
            
            for _ in range(val_input.size(0)):
                x_hat_val = netG(val_target)[0]

            tensor = x_hat_val.data.cpu()

            filename = os.path.join(opt.outdir, str(i) + '.png')

            tensor = torch.squeeze(tensor)
            tensor = norm_range(tensor, None)
            print('Patch:' + str(i))

            ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            im = Image.fromarray(ndarr)
            im.save(filename)

            t1 = time.time()
            print('Running time:' + str(t1-t0))

if __name__ == "__main__":
    main()
