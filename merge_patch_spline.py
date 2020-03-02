"""
  Filename       [ merge_patch_spline.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ]
"""

import os
import sys
from PIL import Image
import numpy as np
import argparse
import time
from scipy.interpolate import CubicSpline
from natsort import natsorted, ns

# 4~5 minutes per 5-patched image

def main():
    CURR_DIR = os.path.abspath(os.path.dirname(__file__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--Hazy_dir', type=str,          # ./indoor/GT_small or ./sample/hazysmall 
        default='./test_images/J_original')
    parser.add_argument('--unprocessed_dir', type=str,   # ./indoor/GT_small or ./sample/hazysmall 
        default='./test_images/result_patches')
    parser.add_argument('--processed_dir', type=str, 
        default='./test_images/result')
    parser.add_argument('--format', type=str,
        default='png', choices=['png', 'tiff', 'jpg'])
    parser.add_argument('--L', type=int, 
        default=2048)
    opt = parser.parse_args()

    Hazy_dir = opt.Hazy_dir
    if not os.path.exists(Hazy_dir):
        os.makedirs(Hazy_dir)

    unprocessed_dir = opt.unprocessed_dir
    if not os.path.exists(unprocessed_dir):
        os.makedirs(unprocessed_dir)

    processed_dir = opt.processed_dir 
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    L = opt.L

    image_index = 0
    size_all = []
    function = []
    ratio = [0] * (L*2)
    Hazyimg1 = []

    for root1, _, fnames in (os.walk(Hazy_dir)):
        for i, fname1 in enumerate( natsorted(fnames, key=lambda y: y.lower()) ):
            img1 = Image.open(os.path.join(CURR_DIR, Hazy_dir, fname1)).convert('RGB') 
            # img = Image.open(os.path.join(root, fname)).convert('RGB')
            Hazyimg1.append(img1)
    

    # f1(x), f2(x), f3(x), f4(x), f5(x) CubicSpline
    # ratio = f1(x) + f2(x) + f3(x) + f4(x) + f5(x)

    for i in range(0, 5): 
        start1 = i * int(L / 4)
        end1   = i * int(L / 4) + L
        # X = np.array([start1, (start1+end1-1)/2, end1-1])
        # Y = np.array([0.1, 1, 0.1]) # !!!!change the 1, 2, 1 to 0.1, 1, 0.1 will improve? YESSSSS!!
        # X = np.array([start1, start1+256-1, end1-1-(256-1), end1-1])
        # Y = np.array([0, 0.1, 0.1, 0])
        
        #------------------------------------------
        X = np.linspace(start1, end1 - 1, num=20)
        X = np.delete(X, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        x = np.arange(5) # 20/2
        y = np.power(x, 2)
        _y = np.flip(y)
        Y = np.concatenate((y, _y), axis=0)
        #------------------------------------------
        function.append(CubicSpline(X,Y))
        
        for C in range(start1, end1):
            ratio[C] += function[i](C) 

    for all_index in range(0, 5):
        image1 = np.zeros((L, L*2, 3, 5))
        t0 = time.time()
        
        for i in range(0,5): 
            img1 = Image.open(os.path.join(CURR_DIR, unprocessed_dir, str(image_index)+'.'+opt.format)).convert('RGB')
           
            image_index = image_index + 1
            w, h = Hazyimg1[i].size
            size_all.append([w, h])
            img1 = img1.resize((L, L), Image.ANTIALIAS)
            img1 = np.asarray(img1)
            start1, end1 = i * int(L/4), i * int(L/4)+L

            # image1(1:1024, start1:end1, 1:3, i) = img1; 
            # relations between PIL/ndarray/matlab array/
            for ch in range(0, 3):
                for R in range(0, L):
                    for C in range(start1,end1):
                        # f1(C) * img1[R][C-start1][ch], f1(x) is spline function y=1~2
                        # when you ratio it you use the region 1 of f1(x) + f2(x) + f3(x) + f4(x) + f5(x) to ratio img1, and then add ratioed img1 to image1
                        if(ratio[C] != 0):
                            image1[R][C][ch][i] = function[i](C) * img1[R][C-start1][ch]
                        else:
                            image1[R][C][ch][i] = img1[R][C-start1][ch]

        zz2 = image1.sum(axis=3)
        for ch in range(0, 3):
            for R in range(0, L):
                for C in range(0, L*2):        
                    if(ratio[C] != 0):                           
                        zz2[R][C][ch] = (zz2[R][C][ch]) / ratio[C] 

        img1 = zz2 / 255
        img1 = Image.fromarray(np.uint8(zz2),'RGB')
        img1 = img1.resize( (size_all[all_index][0], size_all[all_index][1]), Image.ANTIALIAS)

        img1.save(os.path.join(CURR_DIR, processed_dir, str(all_index) + '.' + opt.format))
        t1 = time.time()
        print('Done!, running time: {}s'.format(str(t1 - t0)))

if __name__ == "__main__":
    main()
