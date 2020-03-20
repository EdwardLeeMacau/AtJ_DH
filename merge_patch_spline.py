"""
  Filename       [ merge_patch_spline.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ] 
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline

parser = argparse.ArgumentParser()
parser.add_argument('--ref', required=True,  help='')
parser.add_argument('--patch', required=True,  help='')
parser.add_argument('--merge', required=True, help='')
parser.add_argument('--segment', type=int, default=5)
parser.add_argument('--L', type=int, default=1200, help='')
opt = parser.parse_args()

if not os.path.exists(opt.ref):
    raise ValueError()

if not os.path.exists(opt.patch):
    raise ValueError()

if not os.path.exists(opt.merge):
    os.makedirs(opt.merge)

L = opt.L

def main():
    for i, fname1 in enumerate(os.listdir(opt.ref)):
        img_name = os.path.join(opt.ref, fname1)
        basename, surfix = fname1.split('.')
        orignalImg = Image.open(img_name).convert('RGB')
        w, h = orignalImg.size
        
        image1 = np.zeros((h, w, 3, 5))
        t0 = time.time()

        ratio = np.zeros((w, 5))

        # Make spline coeff.
        for i in range(0, 5): 
            start1= i * int((w - L) / 4)
            end1  = i * int((w - L) / 4) + L
            
            # X = np.array([start1, (start1+end1-1)/2, end1-1])
            # Y = np.array([0.1, 1, 0.1]) # !!!!change the 1, 2, 1 to 0.1, 1, 0.1 will improve? YESSSSS!!
            # X = np.array([start1, start1+256-1, end1-1-(256-1), end1-1])
            # Y = np.array([0, 0.1, 0.1, 0])
            
            #------------------------------------------
            X = np.linspace(start1, end1-1, num=20)
            X = np.delete(X, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            x = np.arange(5) # 20/2
            y = np.power(x, 2)
            _y = np.flip(y)
            Y = np.concatenate((y, _y), axis=0)
            #------------------------------------------
            
            ratio[start1: end1, i] = CubicSpline(X, Y)(np.arange(start1, end1)) 

        # Splining
        for i in range(0, 5):
            img1 = Image.open(os.path.join(opt.patch, basename +'_' + str(i) + '.'+ surfix)).convert('RGB')
            img1 = img1.resize((L, L), Image.ANTIALIAS)
            img1 = np.asarray(img1)
            start1= i * int((w - L) / 4)
            end1  = i * int((w - L) / 4) + L

            mask = np.where(ratio.sum(axis=1)[start1:end1] == 0, 1, ratio[start1: end1, i])
            mask = mask.reshape((1, -1, 1))
            mask = np.tile(mask, (L, 1, 3))

            image1[0:L, start1:end1, :, i] = mask * img1

        ratio = ratio.sum(axis=1)
        ratio = np.where(ratio == 0, 1, ratio).reshape(1, -1, 1)
        ratio = np.tile(ratio, (L, 1, 3))
        
        # Take image
        zz2  = image1.sum(axis=3) / ratio
        img1 = Image.fromarray(np.uint8(zz2),'RGB')
        img1 = img1.resize((w, h), Image.ANTIALIAS)
        img1.save(os.path.join(opt.merge, basename + '.' + surfix))

if __name__ == "__main__":
    main()