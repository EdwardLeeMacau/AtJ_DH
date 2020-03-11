"""
  Filename       [ utils.py ]
  PackageName    [ AtJ_DH.utils ]
  Synopsis       [ ]
"""

import os
import cv2
import numpy as np
from PIL import Image

import torch

def details(opt):
    """
    Show and marked down the training settings

    Parameters
    ----------
    opt : namespace
        The namespace of the train setting    
    """

    for item, values in vars(opt).items():
        print("{:24} {}".format(item, values))
            
    return

def checkdirctexist(dirct):
    """ If directory is not exists, make it. """
    if not os.path.exists(dirct):
        os.makedirs(dirct)

def norm_ip(img, min, max):
    """ Linear normalize **img** """
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img

def norm_range(t, range):
    """ Normalize **t** with parameter **range** """
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
    return norm_ip(t, t.min(), t.max())


def get_image_for_save(img, W, H, pad=6):
    #img=norm_ip(img,img.min(),img.max())
    img = img.data[0].numpy()
    img = img * 255.
    img[img < 0] = 0
    img[img > 255.] = 255.
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    img = img[32*pad:H+32*pad, 32*pad:W+32*pad, :]

    return img

def get_image_for_test(image_name, pad=6):
    img = cv2.imread(image_name)
    img = img.astype(np.float32)
    H, W, C = img.shape
    Wk = W
    Hk = H
    if W % 32:
        Wk = W + (32 * pad - W % 32)
    if H % 32:
        Hk = H + (32 * pad - H % 32)
    img = np.pad(img, ((32*pad, Hk - H), (32*pad, Wk - W), (0, 0)), 'reflect')
    im_input = img / 255.0
    im_input = np.expand_dims(np.rollaxis(im_input, 2), axis=0)

    return im_input, W, H

def my_get_image_for_test(img, pad=6):
    img = img.numpy()
    img = img.astype(np.float32)
    H, W, C = img.shape
    Wk = W
    Hk = H
    if W % 32:
        Wk = W + (32 * pad - W % 32)
    if H % 32:
        Hk = H + (32 * pad - H % 32)
    img = np.pad(img, ((32*pad, Hk - H), (32*pad, Wk - W), (0, 0)), 'reflect')
    im_input = img / 255.0
    # im_input = np.expand_dims(np.rollaxis(im_input, 2), axis=0)
    return  torch.from_numpy(im_input)

def myloader(image_name,pad=6):
    pic = Image.open(image_name).convert('RGB')
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img=my_get_image_for_test(img)
    from IPython import embed
    # embed()
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    
    img=norm(img) 
    img=torch.unsqueeze(img, 0)

    return img

def norm(tensor):  
    mean=(0.5, 0.5, 0.5)
    std=(0.5, 0.5, 0.5)
   
      # TODO: make efficient
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
  
    return tensor
