import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

from natsort import natsorted, ns

# resize to 4096*2048 >> crop patches with size 2048*2048 >> shift the patch 512 every time when done cropping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir', required=True)
    parser.add_argument('--outputDir', required=True)
    parser.add_argument('--w', type=int, default=4096)
    parser.add_argument('--h', type=int, default=2048)
    opt = parser.parse_args()

    if not os.path.isdir(opt.inputDir):
        raise ValueError("Argument --inputDir got wrong value ({})".format(opt.inputDir))

    if not os.path.exists(opt.outputDir):
        os.makedirs(opt.outputDir)

    w = opt.w
    h = opt.h

    for root, _, fnames in (os.walk(opt.inputDir)):
        for i, fname in enumerate( natsorted(fnames, key=lambda y: y.lower()) ):
            print('Filename: {}'.format(fname), end=' ')

            t0 = time.time()

            img = Image.open(os.path.join(opt.inputDir, fname)).convert('RGB')
            img = img.resize((w, h), Image.ANTIALIAS)

            for j in range(0, 5):
                img_t = img.crop((0+(512*j), 0, 2048+(512*j), h))
                img_t.save(os.path.join(opt.outputDir, str(j + 5*i) + '.png'))

            t1 = time.time()

            print('Running time: {}'.format(t1 - t0))

if __name__ == "__main__":
    main()
