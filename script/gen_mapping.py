"""
  Filename      [ gen_mapping.py ]
  PackageName   [ AtJ_DH.artificial_dataset ]
  Synopsis      [  ]
"""

import os 
import json

def main():
    dirs = ['train/I','val/I']

    for dir in dirs :
        dic = {}
        
        for i, j in enumerate(os.listdir(dir)):
            dic[i] = j
        
        with open('{}.json'.format(dir.split("/")[0]), 'w') as f:
            json.dump(dic, f)

if __name__ == "__main__":
    main()