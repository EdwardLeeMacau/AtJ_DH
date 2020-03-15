"""
  Filename       [ dehaze_test.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ]
"""

import argparse
import os
import random
import time
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from PIL import Image

# TODO: Remember change as AtJ_At when train my own model
# import model.AtJ_At as net
import At_model as net
import torchvision
from cmdparser import parser
from misc import *
from utils.utils import norm_ip, norm_range

# cudnn.benchmark = True
# cudnn.fastest = True

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # torch.cuda.set_device(0)

    # CmdParser
    opt = parser.parse_args()

    create_exp_dir(opt.exp)

    # opt.manualSeed = 101
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    opt.dataset = 'pix2pix_val_temp'

    for key, value in vars(opt).items():
        print("{:20} {:>50}".format(key, str(value)))

    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)

    valDataloader = getLoader(
        datasetName=opt.dataset,
        dataroot=opt.valDataroot,
        originalSize=opt.imageSize, #opt.originalSize,
        imageSize=opt.imageSize,
        batchSize=opt.valBatchSize,
        workers=opt.workers,
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

    netG.eval()

    val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
    val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

    # Switch on CUDA
    netG.cuda()

    t0 = time.time()

    with torch.no_grad():
        for i, (data, target, path) in enumerate(valDataloader, 0):
            val_input.resize_as_(data).copy_(data)
            val_target = target.cuda()
            
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
            print('Running time:' + str(t1 - t0))

if __name__ == "__main__":
    main()
