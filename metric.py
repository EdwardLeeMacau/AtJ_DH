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

    for i in range(0, 5):
        GTimg = Image.open(os.path.join(opt.GT_dir, str(i) + '.png'))
        # GTimg = GTimg.resize( (480, 480))
        GTimg = np.array(GTimg)
        
        DHimg = Image.open(os.path.join(opt.DH_dir, str(i) + '.png'))
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
