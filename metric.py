"""
  Filename       [ metric.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ Get inference performance ]
"""

import argparse
import os
from statistics import mean, stdev

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_dir', required=True)
    parser.add_argument('--DH_dir', required=True)
    opt = parser.parse_args()

    psnr_list = []
    ssim_list = []

    GTimgs = [ os.path.join(opt.GT_dir, img_name) for img_name in sorted(os.listdir(opt.GT_dir)) ]
    DHimgs = [ os.path.join(opt.DH_dir, img_name) for img_name in sorted(os.listdir(opt.DH_dir)) ]

    for GT_img_name, DH_img_name in zip(GTimgs, DHimgs):
        GTimg = Image.open(GT_img_name)
        GTimg = np.array(GTimg)
        
        DHimg = Image.open(DH_img_name)
        DHimg = np.array(DHimg)

        psnr = peak_signal_noise_ratio(GTimg, DHimg)
        ssim = structural_similarity(GTimg, DHimg, multichannel=True)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        print('Img {:s} PSNR={:.3f} SSIM={:.5f}'.format(DH_img_name, psnr, ssim))

    print('>> PSNR MEAN={:.3f} STDEV={:.5f}'.format(mean(psnr_list), stdev(psnr_list)))
    print('>> SSIM MEAN={:.3f} STEDV={:.5f}'.format(mean(ssim_list), stdev(ssim_list)))

if __name__ == '__main__':
    main()
