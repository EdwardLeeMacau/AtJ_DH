"""
  Filename       [ psnr_ssim.py ]
  PackageName    [ AtJ_DH ]
"""

import argparse
import os
import statistics

import numpy as np
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim

def main():
    CURR_DIR = os.path.abspath(os.path.dirname(__file__)) #C:/.../Indoor/indoor

    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_dir', required=False, default='./test_images/J_original',  help='')
    parser.add_argument('--DH_dir', required=False, default='./test_images/result',  help='')
    opt = parser.parse_args()

    psnr_list=[]
    ssim_list=[]

    for i in range(0,5):
        GTimg = Image.open(os.path.join(CURR_DIR, opt.GT_dir, str(i) + '.png'))
        GTimg = GTimg.resize( (480, 480))
        GTimg = np.array(GTimg)
        
        DHimg = Image.open(os.path.join(CURR_DIR, opt.GT_dir, str(i) + '.png'))
        DHimg = np.array(DHimg)

        psnr = compare_psnr(GTimg, DHimg)
        ssim = compare_ssim(GTimg, DHimg, multichannel=True)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        print('psnr=%f' % (psnr), end='  ')
        print('ssim=%f' % (ssim))

    print('meanpsnr=%f' % (statistics.mean(psnr_list)) )
    print('meanssim=%f' % (statistics.mean(ssim_list)) )
    print('stdevpsnr=%f' % (statistics.stdev(psnr_list)) )
    print('stdevssim=%f' % (statistics.stdev(ssim_list)) )

if __name__ == '__main__':
    main()