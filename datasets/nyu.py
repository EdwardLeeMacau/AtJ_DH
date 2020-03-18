"""
  Filename       [ nyu.py ]
  PackageName    [ AtJ_DH.datasets ]
  Synopsis       [ ] 
"""

import os

import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile

import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]

def default_loader(path):
    """ PIL Image Loader Wrapper """
    return Image.open(path).convert('RGB')

def np_loader(path):
    """ Numpy Image Loader Wrapper """
    return np.load(path)

def is_image_file(filename):
    """ Return True if the file surfix is an Image. """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class NyuDataset(data.Dataset):
    def __init__(self, root: str, transform=None):
        hazy_path = os.path.join(self.root, 'I')
        gt_path   = os.path.join(self.root, 'J')
        
        self.root = root
        self.hazy_imgs = [ os.path.join(hazy_path, img) for img in os.listdir(hazy_path) ]
        self.gt_imgs   = [ os.path.join(gt_path, img) for img in os.listdir(gt_path) ]
        self.t_matrix  = np.load(os.path.join(self.root, 't.npy'))
        self.a_matrix  = np.load(os.path.join(self.root, 'a.npy'))
        
        self.transform = transform
        
    def __getitem__(self, index):
        """
        Return
        ------
        imgA: 
            Hazy

        imgB: 
            img_A
        
        imgC: 
            img_t

        imgD:
            GT
        """ 
        hazy, gt = self.hazy_imgs[index], self.gt_imgs[index]

        # Hazy, gt
        imgA = Image.open(hazy).convert('RGB')
        imgD = Image.open(gt).convert('RGB')
        
        # A, t
        imgB = self.a_matrix[index]
        imgC = self.t_matrix[index]
        
        # NOTE preprocessing for each pair of images
        if self.transform is not None:
            imgA = self.transform(imgA)
            imgD = self.transform(imgD)

        imgB = np.transpose(imgB, (2, 0, 1)) # hwc to chw
        imgC = np.transpose(imgC, (2, 0, 1)) 

        return imgA, imgB, imgC, imgD

    def __len__(self):
        return len(self.imgs) // 4
