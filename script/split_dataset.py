"""
  Filename      [ split_dataset.py ]
  PackageName   [ AtJ_DH.script ]
  Synopsis      [ Split the dataset into training and validation set ]
"""

import argparse
import os
import random
from shutil import copy

dirs = ['I','J','A','t']
extension = ['.png','.png','.npy','.npy']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, 
        default='../dataset/AtJ_DH/nyu')
    parser.add_argument('--ratio', type=float,
        default=0.8)
    opt = parser.parse_args()

    if not os.path.isdir(opt.directory):
        raise ValueError("Root {} doesn't exist. Please check --directory".format(opt.directory))

    # Generate Number List
    f_list = []
    for i in os.listdir(os.path.join(opt.directory, 'J')):
        f_list.append(i.rstrip('.png').lstrip('J'))
    random.shuffle(f_list)

    # Get the number of training images
    train_num = int(opt.ratio * len(f_list))

    for i, dir in enumerate(dirs):
        # Training data
        dst = os.path.join(opt.directory, 'train', dir)

        if not os.path.exists(dst):
            os.makedirs(dst)

        for index, f in enumerate(f_list[:train_num]):
            fname  = os.path.join(opt.directory, dir, dir + f + extension[i])
            fname2 = os.path.join(dst, '{}_{}{}'.format(dir, index, extension[i]))
            print(fname, fname2)
            copy(fname, fname2)

        # Validation Data
        dst = os.path.join(opt.directory, 'val', dir)

        if not os.path.exists(dst):
            os.makedirs(dst)
        
        for index, f in enumerate(f_list[train_num:]):
            fname  = os.path.join(opt.directory, dir, dir + f + extension[i])
            fname2 = os.path.join(dst, '{}_{}{}'.format(dir, index, extension[i]))
            copy(fname, fname2)

if __name__ == "__main__":
    main()
