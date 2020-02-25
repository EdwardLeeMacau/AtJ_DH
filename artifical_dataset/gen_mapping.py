import os 
import json

dirs=['train/I','val/I']

for dir in dirs :
    f_list=os.listdir(dir)
    dic={}
    for i,j in enumerate(f_list):
        dic[i]=j
    with open(f'{dir.split("/")[0]}.json','w') as f:
        json.dump(dic,f)