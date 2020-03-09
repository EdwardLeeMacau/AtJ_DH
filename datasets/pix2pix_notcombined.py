"""
  Filename       [ pix2pix_notcombined.py ]
  PackageName    [ AtJ_DH.datasets ]
  Synopsis       [ ] 
"""

import os
import os.path

import numpy as np
from PIL import Image, ImageFile

import torch.utils.data as data
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
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if not os.path.isdir(dir):
        raise Exception('Check dataroot')

    images = []
 
    for _, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                item = path
                images.append(item)

    return images

class pix2pix_notcombined(data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader, seed=None):
        imgs = make_dataset(root)

        if len(imgs) == 0:
            raise ValueError(
                "Found 0 images in subfolders of: " + root + "\n" + 
                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
            )

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        
        if seed is not None:
            np.random.seed(seed)

    def __getitem__(self, index):
        """
        Return
        ------
        imgA: 
            hazy

        imgB: 
            img_A
        
        imgC: 
            img_t
        """ 
        hazy_path = self.root + '/I/I_' + str(index) + '.png'
        gt_path   = self.root + '/J/J_' + str(index) + '.png'
        A_path    = self.root + '/A/A_' + str(index) + '.npy'
        t_path    = self.root + '/t/t_' + str(index) + '.npy'

        imgA = self.loader(hazy_path)
        imgB = np_loader(A_path)
        imgC = np_loader(t_path)
        imgD = self.loader(gt_path)
        
        # NOTE preprocessing for each pair of images
        if self.transform is not None:
            imgA = self.transform(imgA)
            imgD = self.transform(imgD)

        imgB = np.transpose(imgB, (2, 0, 1)) # hwc to chw
        imgC = np.transpose(imgC, (2, 0, 1)) 

        return imgA, imgB, imgC, imgD

    def __len__(self):
        return len(self.imgs) // 4
