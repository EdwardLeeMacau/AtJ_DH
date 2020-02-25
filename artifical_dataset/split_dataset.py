import os
import random
from shutil import copy
f_list = []
for i in os.listdir('J'):
    f_list.append(i.rstrip('.png').lstrip('J'))
# print(f_list[:30])
random.shuffle(f_list)
dirs=['I','J','A','t']
extension=['.png','.png','.npy','.npy']

train_ratio=0.8
train_num=int(train_ratio*len(f_list))
## train data
for i,dir in enumerate(dirs):
    dst='train/'+dir+'/'

    if not os.path.exists(dst):
        os.mkdir('train/'+dir)
    for index,f in enumerate(f_list[:train_num]):
        fname=dir+'/'+dir+f+extension[i]
        print(fname)
        copy(fname,f'{dst}{dir}_{index}{extension[i]}')
    dst='val/'+dir+'/'
    if not os.path.exists(dst):
        os.mkdir('val/'+dir)
    for index,f in enumerate(f_list[train_num:]):
        fname=dir+'/'+dir+f+extension[i]
        print(fname)
        copy(fname,f'{dst}{dir}_{index}{extension[i]}')

