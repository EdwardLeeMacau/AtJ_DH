"""
  Filename       [ MAINTHREAD_test.py ]
  PackageName    [ AtJ_DH ]
"""

import os
import argparse
import time
import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from utils import utils
# import model.AtJ_At as net
from model.At_model import Dense
# from At_model_feature import Dense
from utils.utils import norm_ip, norm_range

def saveImage(tensor, H, W, pad, fname):
    tensor = torch.squeeze(tensor)

    # tensor = norm_range(tensor, None)
    print(tensor.min(), tensor.max(), fname)
    tensor = tensor[ :, 32*pad: 32*pad+H, 32*pad: 32*pad+W ]
    tensor = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0)

    ndarr  = tensor.numpy()
    ndarr  = ndarr[:, :, ::-1]

    im = Image.fromarray(ndarr)
    im.save(fname)

    return


def main():
    print("Testing model ........ ")

    parser = argparse.ArgumentParser(description="Pytorch AtJ_model Evaluation")
    parser.add_argument("--cuda", action="store_true", 
        help="use cuda? Default is True")
    parser.add_argument("--model", type=str, 
        default="./model/At_model2.pth")
    parser.add_argument("--test", type=str, 
        default="./test_images/NTIRE2019_RAW/test/Hazy", help="testset path")
    parser.add_argument('--outdir', type=str,
        default='./test_images/NTIRE2019_RAW/test/DeHazy', help='path to output folder')
    parser.add_argument('--gt', type=str,
        default='./test_images/NTIRE2019_RAW/test/GT', help='path to GT folder')
    parser.add_argument('--rehaze', type=str,
        default='./test_images/NTIRE2019_RAW/test/ReHazy', help='path to output folder')
    opt = parser.parse_args()

    cuda = opt.cuda
    device_label = 'GPU' if opt.cuda else 'CPU'

    if cuda and not torch.cuda.is_available():
        raise Exception(">>No GPU found, please run without --cuda")

    if not cuda:
        print(">>Run on *CPU*, the running time will be longer than reported GPU run time. \n"
              ">>To run on GPU, please run the script with --cuda option")

    utils.checkdirctexist(opt.outdir)
    utils.checkdirctexist(opt.rehaze)

    # model = net.AtJ()
    model = Dense()
    model.load_state_dict(torch.load(opt.model))
    # model = torch.load(opt.model)["model"]
    model.eval()

    if cuda:
        model.cuda()

    image_list = glob.glob(os.path.join(opt.test, '*.png'))

    print(">>Start testing...\n"
        "\t\t Model: {0}\n"
        "\t\t Test on: {1}\n"
        "\t\t Results save in: {2}".format(opt.model, opt.test, opt.outdir))

    avg_elapsed_time = 0.0
    i = 0
    pad = 6

    with torch.no_grad():
        # DeHaze
        image_list = glob.glob(os.path.join(opt.test, '*.png'))

        for image_name in sorted(image_list):
            print(">> Processing ./{}".format(image_name))
            im_input, W, H = utils.get_image_for_test(image_name, pad=pad)

            im_input = torch.from_numpy(im_input).float()

            if cuda:
                im_input = im_input.cuda()
            else:
                im_input = im_input.cpu()
            
            start_time = time.time()

            # feeding forward
            im_output = model(im_input)
            im_output, A, t, _ = im_output
            
            # compute running time
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            tensor = im_output.data.cpu()
            saveImage(torch.squeeze(tensor), H, W, pad, os.path.join(opt.outdir, str(i) + '.png'))

            tensor = A.data.cpu()
            saveImage(torch.squeeze(tensor), H, W, pad, os.path.join(opt.outdir, str(i) + '_a.png'))

            tensor = t.data.cpu()
            saveImage(torch.squeeze(tensor), H, W, pad, os.path.join(opt.outdir, str(i) + '_t.png'))

            i += 1

        # ReHaze
        image_list = glob.glob(os.path.join(opt.gt, '*.png'))
        i = 0
        for image_name in sorted(image_list):
            print(">> Processing ./{}".format(image_name))
            im_input, W, H = utils.get_image_for_test(image_name, pad=pad)

            im_input = torch.from_numpy(im_input).float()

            if cuda:
                im_input = im_input.cuda()
            else:
                im_input = im_input.cpu()

            start_time = time.time()

            # feeding forward
            im_output = model(im_input)
            _, A, t, im_output = im_output
            
            # compute running time
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            tensor = im_output.data.cpu()
            saveImage(torch.squeeze(tensor), H, W, pad, os.path.join(opt.rehaze, str(i) + '.png'))

            tensor = A.data.cpu()
            saveImage(torch.squeeze(tensor), H, W, pad, os.path.join(opt.rehaze, str(i) + '_a.png'))

            tensor = t.data.cpu()
            saveImage(torch.squeeze(tensor), H, W, pad, os.path.join(opt.rehaze, str(i) + '_t.png'))

            i += 1

    print(">> Finished!")
    print("It takes average {}s for processing single image on {}".format(avg_elapsed_time / len(image_list), device_label))
    print("Results are saved at {}".format(opt.outdir))

if __name__ == "__main__":
    main()
