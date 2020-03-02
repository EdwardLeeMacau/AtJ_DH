"""
  Filename       [ merge_patch.py ]
  PackageName    [ AtJ_DH ]
  Synopsis       [ ] 
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image


def main():
    CURR_DIR = os.path.abspath(os.path.dirname(__file__)) #C:/.../Indoor
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_dir', required=False, # ./indoor/GT_small or ./sample/hazysmall
        default='./test_images/J_original',  help='')
    # parser.add_argument('--GT_dir', required=False, # ./indoor/GT_small or ./sample/hazysmall
    #    default='./test_images/I_space_png',  help='')
    parser.add_argument('--unprocessed_dir', required=False, # ./indoor/GT_small or ./sample/hazysmall
        default='./test_images/result_patches',  help='')
    parser.add_argument('--processed_dir', required=False, 
        default='./test_images/result',  help='')
    opt = parser.parse_args()

    GT_dir = opt.GT_dir
    if not os.path.exists(GT_dir):
        os.makedirs(GT_dir)

    unprocessed_dir = opt.unprocessed_dir
    if not os.path.exists(unprocessed_dir):
        os.makedirs(unprocessed_dir)

    processed_dir = opt.processed_dir 
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    
    image_index=0
    size_all=[]

    for all_index in range(0,5):
        image1 = np.zeros((1024,2048,3,5))
        t0 = time.time()

        for i in range(0,5): 
            img1 = Image.open(os.path.join(CURR_DIR, unprocessed_dir, str(image_index)+'.png')).convert('RGB')
            GTimg1 = Image.open(os.path.join(CURR_DIR, GT_dir,str(i)+'.png')) # Indoor GT_test name is 31~35!!!!
            w, h = GTimg1.size
            
            size_all.append([w, h])
            img1 = img1.resize((1024, 1024), Image.ANTIALIAS)
            img1 = np.asarray(img1)
            image_index=image_index + 1
            start1=i*256
            end1=i*256+1024

            for ch in range(0,3): # image1(1:1024,start1:end1,1:3,i)=img1; relations between PIL/ndarray/matlab array/
                for R in range(0,1024):
                    for C in range(start1,end1):
                        image1[R][C][ch][i]=img1[R][C-start1][ch]
                        
                                    
        zz2=image1.sum(axis=3)
        # matlab: zz2=sum(image1,4)
        
        ratio=[1,2,3,4,4,3,2,1]

        zz3=np.zeros((1024,2048,3,8)) # must have nested parenthesis

        zz4=np.zeros((1024,256,3))

        for index in range(0,8):
            start2=index*256
            end2=index*256+256
            for ch in range(0,3):
                for R in range(0,1024):
                    for C in range(start2,end2):
                        zz4[R][C-start2][ch]=(zz2[R][C][ch])/ratio[index]
            # matlab: zz4=zz2[0:1024,start2:end2,0:3]/ratio[index]
            for ch in range(0,3):
                for R in range(0,1024):
                    for C in range(start2,end2):
                        zz3[R][C][ch][index]=zz4[R][C-start2][ch]
            # matlab: zz3[0:1024,start2:end2,0:3,index]=zz4

        zz4=zz3.sum(axis=3)
        img1=zz4/255
        img1=Image.fromarray(np.uint8(zz4),'RGB') # can't just do np.(zz4) only!!!!
        img1=img1.resize( (size_all[all_index][0], size_all[all_index][1]), Image.ANTIALIAS)

        img1.save(os.path.join(CURR_DIR, processed_dir, str(all_index)+'.png'))
        print('Done!')
        t1 = time.time()
        print('running time:'+str(t1-t0))

if __name__ == "__main__":
    main()