"""
  Filename      [ gen_mapping.py ]
  PackageName   [ AtJ_DH.artificial_dataset ]
  Synopsis      [  ]
"""

import os 
import json

dirs = ['train/I','val/I']

for dir in dirs :
    dic = {}
    
    for i,j in enumerate(os.listdir(dir)):
        dic[i]=j
    
    with open(f'{dir.split("/")[0]}.json','w') as f:
        json.dump(dic,f)