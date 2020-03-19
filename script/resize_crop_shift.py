import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

# resize to 4096*2048 >> crop patches with size 2048*2048 >> shift the patch 512 every time when done cropping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir', required=True)
    parser.add_argument('--outputDir', required=True)
    parser.add_argument('--resize', type=int)
    parser.add_argument('--segment', type=int, default=5)
    parser.add_argument('--direction', type=str, choices=[None, 'Horizontal', 'Vertical'])
    opt = parser.parse_args()

    if not os.path.isdir(opt.inputDir):
        raise ValueError("Argument --inputDir got wrong value ({})".format(opt.inputDir))

    if not os.path.exists(opt.outputDir):
        os.makedirs(opt.outputDir)

    for fname in (os.listdir(opt.inputDir)):
        basename, surfix = fname.split('.')
        t0 = time.time()

        img  = Image.open(os.path.join(opt.inputDir, fname)).convert('RGB')

        if opt.direction is not None:
            raise NotImplementedError

        if opt.direction is None:
            H, W = img.size
            unit = min(H, W)
            step = int((max(H, W) - unit) / opt.segment)
            opt.direction = 'Horizontal' if (W > H) else "Vertical"

        # If loss pixels
        if opt.step * opt.segment + unit != max(H, W):
            raise NotImplementedError

        for j in range(opt.segment):
            if opt.direction == "Horizontal":
                img_t = img.crop(((step * j), 0, unit + (step * j), H))
            if opt.direction == "Vertical":
                img_t = img.crop((0, step * j, W, unit + step * j))
            
            if opt.resize is not None:
                img_t = img_t.resize((opt.resize, opt.resize))

            img_t.save(os.path.join(opt.outputDir, basename + '_' + str(j)+ surfix))

        t1 = time.time()
        print('>> [{:.2f}] Filename: {}'.format(t1 - t0, os.path.join(opt.inputDir, fname)))
        
if __name__ == "__main__":
    main()
