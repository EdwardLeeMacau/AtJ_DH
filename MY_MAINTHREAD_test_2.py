"""
  Filename       [ MY_MAINTHREAD_test.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ]
"""

import argparse
import os
import time

import numpy as np
import torch

import model.AtJ_At as atj
from datasets.data import ValidationDatasetFromFolder
from misc_train import *
from model.At_model import Dense
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from utils.utils import norm_ip, norm_range

from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netG', required=True)
    parser.add_argument('--haze', required=True)
    parser.add_argument('--dehaze', required=True)
    parser.add_argument('--parse')
    parser.add_argument('--rehaze')
    opt = parser.parse_args()

    if opt.netG is None:
        raise ValueError("netG should not be None.")

    if not os.path.exists(opt.netG):
        raise ValueError("netG {} doesn't exist".format(opt.netG))

    for key, value in vars(opt).items():
        print("{:20} {:>50}".format(key, str(value)))

    img_transform = Compose([
        Resize((512, 1024)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valDataloader = DataLoader(
        dataset=ValidationDatasetFromFolder(opt.haze, transform=img_transform), 
        num_workers=8, 
        batch_size=1, 
        pin_memory=True, 
        shuffle=False
    )

    model = Dense()
    model.load_state_dict(torch.load(opt.netG)['model'])
    model.eval()
    model.cuda()

    # Main Loop of training
    t0 = time.time()

    with torch.no_grad():
        for i, (data, fname) in enumerate(valDataloader, 1):
            data  = data.float().cuda()
            fname = os.path.join(opt.dehaze, os.path.basename(fname[0]))

            output = model(data)[0]
            output = output.cpu()

            output.mul_(torch.Tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)).add_(torch.Tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1))
            
            output = torch.squeeze(output)
            output = norm_range(output, None)
            output = output.mul(255).byte().numpy()
            output = np.transpose(output, (1, 2, 0)) # CHW to HWC

            im = Image.fromarray(output)
            im = im.resize((1600, 1200))
            im.save(fname)

            # Show Message
            print('>> [{:.2f}] [{:3d}/{:3d}] Saved: {}'.format(
                time.time() - t0, i, len(valDataloader), fname))
            
if __name__ == '__main__':
    main()