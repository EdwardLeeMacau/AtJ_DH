"""
  Filename      [ split_dataset.py ]
  PackageName   [ AtJ_DH.artificial_dataset ]
  Synopsis      [ Split the dataset into training and validation set ]
"""

import argparser
import os
import random
from shutil import copy

dirs = ['I','J','A','t']
extension = ['.png','.png','.npy','.npy']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, 
        default='../dataset/AtJ_DH/nyu')
    parser.add_argument('--ratio', type=float
        default=0.8)
    opt = parser.parse_args()

    if not os.path.isdir(opt.directory):
        raise ValueError("Root {} doesn't exist. Please check --directory".format(opt.directory))

    # Generate Number List
    f_list = []
    for i in os.listdir('J'):
        f_list.append(i.rstrip('.png').lstrip('J'))
    random.shuffle(f_list)

    # Get the number of training images
    train_num = int(opt.ratio * len(f_list))

    for i, dir in enumerate(dirs):
        # Training data
        dst = os.path.join('train', dir)

        if not os.path.exists(dst):
            os.mkdir(dst)

        for index, f in enumerate(f_list[:train_num]):
            fname = os.path.join(dir, dir + f + extension[i])
            print(fname)
            copy(fname, f'{dst}{dir}_{index}{extension[i]}')
        
        dst = os.path.join('val', dir)

        # Validation Data
        if not os.path.exists(dst):
            os.mkdir(dst)
        
        for index, f in enumerate(f_list[train_num:]):
            fname = dir + '/' + dir + f + extension[i]
            print(fname)
            copy(fname,f'{dst}{dir}_{index}{extension[i]}')

if __name__ == "__main__":
    main()
