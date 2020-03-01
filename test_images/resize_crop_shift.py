import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

from natsort import natsorted, ns

# resize to 4096*2048 >> crop patches with size 2048*2048 >> shift the patch 512 every time when done cropping

CURR_DIR = os.path.abspath(os.path.dirname(__file__)) #C:/.../Indoor
parser = argparse.ArgumentParser()
parser.add_argument('--unprocessed_dir', required=False, # ./indoor/GT_small or ./sample/hazysmall
  default='',  help='')
parser.add_argument('--processed_dir', required=False, 
  default='',  help='')
parser.add_argument('--w', required=False, type=int,
  default=4096,  help='')
parser.add_argument('--h', required=False, type=int,
  default=2048,  help='')
opt = parser.parse_args()

unprocessed_dir = opt.unprocessed_dir
if not os.path.exists(unprocessed_dir):
  os.makedirs(unprocessed_dir)

processed_dir = opt.processed_dir 
if not os.path.exists(processed_dir):
  os.makedirs(processed_dir)

w = opt.w
h = opt.h


for root, _, fnames in (os.walk(unprocessed_dir)):
  for i, fname in enumerate( natsorted(fnames, key=lambda y: y.lower()) ):
    t0 = time.time()
    img = Image.open(os.path.join(CURR_DIR, unprocessed_dir, fname)).convert('RGB') # img = Image.open(os.path.join(root, fname)).convert('RGB')
    img = img.resize((w, h), Image.ANTIALIAS)

    # crop 2048*2048 per patch, shift right 512 per patch
    for j in range(0,5):
      img_t = img.crop((0+(512*j), 0, 2048+(512*j), h))
      img_t.save(os.path.join(CURR_DIR, processed_dir, str(j + 5*i) + '.png'))
    t1 = time.time()
    print('running time:'+str(t1-t0))
