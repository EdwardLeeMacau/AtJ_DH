"""
  Filename      [ split_dataset.py ]
  PackageName   [ AtJ_DH.artificial_dataset ]
  Synopsis      [ Split the dataset into training and validation set ]
"""

import os
import random
from shutil import copy

dirs = ['I','J','A','t']
extension = ['.png','.png','.npy','.npy']
    
def main():
    f_list = []
    for i in os.listdir('J'):
        f_list.append(i.rstrip('.png').lstrip('J'))
    # print(f_list[:30])
    random.shuffle(f_list)

    train_ratio=0.8
    train_num=int(train_ratio*len(f_list))

    ## train data
    for i, dir in enumerate(dirs):
        dst = os.path.join('train', dir)

        if not os.path.exists(dst):
            os.mkdir(dst)
        for index, f in enumerate(f_list[:train_num]):
            fname = dir+'/'+dir+f+extension[i]
            print(fname)
            copy(fname,f'{dst}{dir}_{index}{extension[i]}')
        
        dst = os.path.join('val', dir)

        if not os.path.exists(dst):
            os.mkdir(dst)
        
        for index,f in enumerate(f_list[train_num:]):
            fname = dir + '/' + dir + f + extension[i]
            print(fname)
            copy(fname,f'{dst}{dir}_{index}{extension[i]}')

if __name__ == "__main__":
    main()
