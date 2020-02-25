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
#cudnn.benchmark = True
#cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from misc import *
from torch.autograd import Variable

# import models.dehaze1113 as net
#import dehaze1113 as net
# import model.AtJ_At as net


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

# import sys
# sys.setrecursionlimit(10000)



def main():
    print("dehazetest.py start!!")
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False,
        default='pix2pix',  help='')
    parser.add_argument('--dataroot', required=False,
        default='/mnt/dehaze/dehazing/AtJ_DH_final/test_images/I_ori_patches', help='path to train dataset')
    parser.add_argument('--valDataroot', required=False,
        default='/mnt/dehaze/dehazing/AtJ_DH_final/test_images/I_ori_patches', help='path to val dataset')
    parser.add_argument('--outdir', required=False,
        default='./test_images/result_patches/', help='path to output folder')
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
    parser.add_argument('--lambdaGAN', type=float, default=0.01, help='lambdaGAN')
    parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
    parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
    parser.add_argument('--netG', default='./pretrained-model/finalCKPT_at.pth', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
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

    opt.dataset='pix2pix_val_temp'

    valDataloader = getLoader(opt.dataset,
                            opt.valDataroot,
                            opt.imageSize, #opt.originalSize,
                            opt.imageSize,
                            opt.valBatchSize,
                            opt.workers,
                            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                            split='val',
                            shuffle=False,
                            seed=opt.manualSeed)

    inputChannelSize = opt.inputChannelSize
    outputChannelSize= opt.outputChannelSize

    #netG=net.Dense_rain_cvprw3()
    # netG.apply(weights_init)
    netG=net.AtJ()

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

    val_target= torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
    val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

    val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
    val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

    # netG.cuda()
    # criterionBCE.cuda()
    # criterionCAE.cuda()

    # val_target, val_input = val_target.cuda(), val_input.cuda()


    for epoch in range(1):
        for i, data_val in enumerate(valDataloader, 0):
            if 1:
                t0 = time.time()

            val_input_cpu, val_target_cpu, path = data_val

            # val_target_cpu, val_input_cpu = val_target_cpu.float().cuda(), val_input_cpu.float().cuda()

            val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
            val_target=Variable(val_target_cpu, volatile=True)
            
            for idx in range(val_input.size(0)):
                x_hat_val = netG(val_target)[0]
                # from IPython import embed
                # embed()

                # val_batch_output[idx,:,:,:].copy_(x_hat_val.data)
            # vutils.save_image(x_hat_val.data, './image_heavy/'+str(i)+'.jpg', normalize=True, scale_each=False,  padding=0, nrow=1)
            tensor = x_hat_val.data.cpu()

            directory = opt.outdir
            if not os.path.exists(directory):
                os.makedirs(directory)

            name = ''.join(path)
            filename = directory+'/' + str(i) + '.png'

            tensor = torch.squeeze(tensor)
            tensor=norm_range(tensor, None)
            print('Patch:'+str(i))


            ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            im = Image.fromarray(ndarr)
            im.save(filename)
            t1 = time.time()
            print('running time:'+str(t1-t0))

if __name__ == "__main__":
    main()
