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
    parser.add_argument('--rehaze', type=str, 
        help='path to output folder')
    opt = parser.parse_args()

    device_label = 'GPU' if opt.cuda else 'CPU'

    if opt.cuda and not torch.cuda.is_available():
        raise Exception(">>No GPU found, please run without --cuda")

    if not opt.cuda:
        print(">>Run on *CPU*, the running time will be longer than reported GPU run time. \n"
              ">>To run on GPU, please run the script with --cuda option")

    utils.checkdirctexist(opt.outdir)
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

    # DeHaze
    with torch.no_grad():
        haze_imgs = sorted(glob.glob(os.path.join(opt.test, '*.png')))

        for i, haze_img in enumerate(haze_imgs):
            print(">> Processing {}".format(haze_img))

            # Data Preparing
            haze, W, H = utils.get_image_for_test(haze_img, pad=pad)
            haze = torch.from_numpy(haze).float()
            haze = haze.cuda() if opt.cuda else haze.cpu()
            
            # compute running time
            start_time = time.time()

            # feeding forward
            dehaze, A, t = model(haze)
            
            # compute running time
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            # Save Output to new directory
            haze_img = os.path.basename(haze_img)

            tensor = dehaze.data.cpu()
            saveImage(tensor, H, W, pad, os.path.join(opt.outdir, haze_img))

            tensor = A.data.cpu()
            saveImage(tensor, H, W, pad, os.path.join(opt.outdir, haze_img.split('.')[0] + '_a.png'))

            tensor = t.data.cpu()
            saveImage(tensor, H, W, pad, os.path.join(opt.outdir, haze_img.split('.')[0] + '_t.png'))

    # Dehaze time count
    print(">> Finished!")
    print("It takes average {}s for processing single image on {}".format(
        avg_elapsed_time / len(gt_imgs), device_label))
    print("Results are saved at {}".format(opt.outdir))

    if opt.gt is None:
        print("No ground truth images are provided, can't rehaze images. ")
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

            # tensor = A.data.cpu()
            # saveImage(tensor, H, W, pad, os.path.join(opt.rehaze, haze_img.split('.')[0] + '_a.png'))

            # tensor = t.data.cpu()
            # saveImage(tensor, H, W, pad, os.path.join(opt.rehaze, haze_img.split('.')[0] + '_t.png'))

if __name__ == "__main__":
    main()
