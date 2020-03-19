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

    # TODO
    if opt.direction is not None:
        raise NotImplementedError


    for fname in (os.listdir(opt.inputDir)):
        basename, surfix = fname.split('.')
        t0 = time.time()

        img  = Image.open(os.path.join(opt.inputDir, fname)).convert('RGB')
        print('>>> Filename: {}'.format(os.path.join(opt.inputDir, fname)))
        print('>>>     Raw Shape(W x H): ({:d}, {:d})'.format(*img.size))

        if opt.direction is None:
            W, H = img.size
            unit = min(H, W)
            step = int((max(H, W) - unit) / (opt.segment - 1))
            opt.direction = 'Horizontal' if (W > H) else "Vertical"

        # If loss pixels
        if step * (opt.segment - 1) + unit != max(H, W):
            raise NotImplementedError

        for j in range(opt.segment):
            # Get the crop range
            if opt.direction == "Horizontal":
                location = ((step * j), 0, unit + (step * j), H)
            if opt.direction == "Vertical":
                location = ((0, step * j, W, unit + step * j))

            img_t = img.crop(location)
            
            if opt.resize is not None:
                img_t = img_t.resize((opt.resize, opt.resize))

            img_t.save(os.path.join(opt.outputDir, basename + '_' + str(j) + '.' + surfix))
            print('>>>     Saved: {}'.format(os.path.join(opt.outputDir, fname)))
            print('>>>     Range: ({:d}, {:d}), ({:d}, {:d})'.format(*location))
            print('>>>     Shape: ({:d}, {:d})'.format(*img_t.size))
 
    t1 = time.time()
       
if __name__ == "__main__":
    main()
