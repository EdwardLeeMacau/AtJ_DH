"""
  Filename       [ metric.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ]
"""

import argparse
import os
from statistics import mean, stdev

import numpy as np
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim

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

        psnr = compare_psnr(GTimg, DHimg)
        ssim = compare_ssim(GTimg, DHimg, multichannel=True)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        print('psnr=%f' % (psnr), end='  ')
        print('ssim=%f' % (ssim))

    print('meanpsnr=%f' % (mean(psnr_list)) )
    print('meanssim=%f' % (mean(ssim_list)) )
    print('stdevpsnr=%f' % (stdev(psnr_list)) )
    print('stdevssim=%f' % (stdev(ssim_list)) )

if __name__ == '__main__':
    main()
