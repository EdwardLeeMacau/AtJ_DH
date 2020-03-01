"""
  Filename      [ gen_mapping.py ]
  PackageName   [ AtJ_DH.artificial_dataset ]
  Synopsis      [  ]
"""

import argparser
import os 
import json

def main():
    parser = argparser.ArgumentParser()
    parser.add_argument("--directory", type=str, default='../dataset/AtJ_DH/nyu')

    for subfolder in ('train/I','val/I'):
        dir = os.path.join(opt.directory, subfolder, 'I')
        dic = {}
        
        for i, j in enumerate(os.listdir(dir)):
            dic[i] = j
        
        with open('{}.json'.format(dir.split("/")[0]), 'w') as f:
            json.dump(dic, f)

if __name__ == "__main__":
    main()
