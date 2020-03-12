"""
  Filename       [ MAINTHREAD_test.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ Test script provided by author ]
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from PIL import Image

import cv2
# import model.AtJ_At as net
from model.At_model import Dense
from utils import utils
# from At_model_feature import Dense
from utils.utils import norm_ip, norm_range
from torchvision.transforms import Compose, Normalize, ToTensor

def saveImage(tensor, H, W, pad, fname):
    tensor = torch.squeeze(tensor)

    # tensor = norm_range(tensor, None)

    # Crop Image and Scale to 0 - 255
    tensor = tensor[ :, 32*pad: 32*pad+H, 32*pad: 32*pad+W ]
    tensor = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0)

    # Reverse RGB Domain
    ndarr  = tensor.numpy()
    ndarr  = ndarr[:, :, ::-1]

    im = Image.fromarray(ndarr)
    im.save(fname)

    return


def main():
    parser = argparse.ArgumentParser(
        description="Pytorch AtJ_model Evaluation")
    parser.add_argument("--cuda", action="store_true", 
        help="use cuda?")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test", type=str, required=True, 
        help="testset path")
    parser.add_argument('--outdir', type=str, required=True, 
        help='path to output folder')
    parser.add_argument('--gt', type=str, 
        help='path to GT folder')
    parser.add_argument('--parse', type=str,
        help='path to A(x), t(x) folder')
    parser.add_argument('--rehaze', type=str, 
        help='path to output folder')
    parser.add_argument('--normalize', action='store_true')
    opt = parser.parse_args()

    device_label = 'GPU' if opt.cuda else 'CPU'

    if opt.cuda and not torch.cuda.is_available():
        raise Exception(">>No GPU found, please run without --cuda")

    if not opt.cuda:
        print(">>Run on *CPU*, the running time will be longer than reported GPU run time. \n"
              ">>To run on GPU, please run the script with --cuda option")

    utils.checkdirctexist(opt.outdir)
    if opt.parse is not None:
        utils.checkdirctexist(opt.parse)
    if opt.rehaze is not None:
        utils.checkdirctexist(opt.rehaze)

    # model = net.AtJ()
    model = Dense()
    model.load_state_dict(torch.load(opt.model))
    model.eval()

    if opt.cuda:
        model.cuda()

    gt_imgs = glob.glob(os.path.join(opt.test, '*.png'))

    print(">>Start testing...\n"
        "\t Model: {0}\n"
        "\t Test on: {1}\n"
        "\t Results save in: {2}".format(opt.model, opt.test, opt.outdir))

    avg_elapsed_time = 0.0
    pad = 6

    img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # DeHaze
    with torch.no_grad():
        haze_imgs = sorted(glob.glob(os.path.join(opt.test, '*.png')))

        for i, haze_img in enumerate(haze_imgs):
            print(">> Processing {}".format(haze_img))

            # Data Preparing
            haze, W, H = utils.get_image_for_test(haze_img, pad=pad)
            
            if not opt.normalize:
                haze = torch.from_numpy(haze).float()
            
            if opt.normalize:
                haze = np.rollaxis(np.squeeze(haze, axis=0), 0, 3)
                haze = img_transform(haze)
                haze = torch.unsqueeze(haze, dim=0)
            
            haze = haze.cuda() if opt.cuda else haze.cpu()
            
            # compute running time
            start_time = time.time()

            # feeding forward
            dehaze, A, t = model(haze)

            # Take it out
            dehaze, A, t = dehaze.cpu(), A.cpu(), t.cpu()
            
            if opt.normalize:
                dehaze.mul_(torch.Tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)).add_(torch.Tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1))
            
            # compute running time
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            # Save Output to new directory
            dehaze_img = os.path.basename(haze_img)

            saveImage(dehaze, H, W, pad, os.path.join(opt.outdir, dehaze_img))
            saveImage(A, H, W, pad, os.path.join(opt.parse, dehaze_img.split('.')[0] + '_a.png'))
            saveImage(t, H, W, pad, os.path.join(opt.parse, dehaze_img.split('.')[0] + '_t.png'))
            saveImage(A * (1 - t), H, W, pad, os.path.join(opt.parse, dehaze_img.split('.')[0] + '_sub.png'))

    # Dehaze time count
    print(">> Finished!")
    print(">> It takes average {}s for processing single image on {}".format(
        avg_elapsed_time / len(gt_imgs), device_label))
    print(">> Results are saved at {}".format(opt.outdir))

    if opt.gt is None:
        print(">> No ground truth images are provided, can't rehaze images. ")
        return

    # ReHaze
    with torch.no_grad():
        gt_imgs = sorted(glob.glob(os.path.join(opt.gt, '*.png')))

        for i, (gt_img, haze_img) in enumerate(zip(gt_imgs, haze_imgs)):
            print(">> Processing {}".format(gt_img))

            # Data Preparing
            gt_img, W, H = utils.get_image_for_test(gt_img, pad=pad)
            gt_img = torch.from_numpy(gt_img).float()
            gt_img = gt_img.cuda() if opt.cuda else gt_img.cpu()

            haze_img, W, H = utils.get_image_for_test(haze_img, pad=pad)
            haze_img = torch.from_numpy(haze_img).float()
            haze_img = haze_img.cuda() if opt.cuda else haze_img.cpu()

            # feeding forward
            _, A, t = model(haze_img)

            tensor = gt_img * t + A * (1 - t)
            saveImage(tensor, H, W, pad, os.path.join(opt.rehaze, haze_img))

if __name__ == "__main__":
    main()
