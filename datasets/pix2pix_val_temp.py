
import os
import os.path

import numpy as np
import torch.utils.data as data
from PIL import Image

# from guidedfilter import guidedfilter

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    """
    :return images: list of image fnames
    """
    images = []
    
    if not os.path.isdir(dir):
        raise Exception('Check dataroot')

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                images.append(os.path.join(dir, fname))
    
    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

class pix2pix_val_temp(data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader, seed=None):
        imgs = make_dataset(root)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                      "Supported image extensions are: " + ", ".join(IMG_EXTENSIONS)))
        
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

        if seed is not None:
            np.random.seed(seed)

    def __getitem__(self, index):
        path = self.imgs[index]

        path = self.root + '/' + str(index) + '.png'

        # index_folder = np.random.randint(0,4)
        # label = index_folder

        img = self.loader(path)

        w, h = img.size
        w, h = 2560, 2560 
        # w, h = 2560*2, 2560
        # w, h = 2560*2, 1280
        # w, h = 2048*4, 2048
        # w, h = 1024*2, 1024

        img = img.resize((w, h), Image.BILINEAR)

        # NOTE: split a sample into imgA and imgB
        imgA = img.crop((0, 0, w, h))
        imgB = img.crop((0, 0, w, h))
        # imgC = img.crop((2*w/3, 0, w, h))
        
        if self.transform is not None:
            # NOTE preprocessing for each pair of images
            # imgA, imgB, imgC = self.transform(imgA, imgB, imgC)
            imgA, imgB = self.transform(imgA, imgB)

        return imgA, imgB, path

    def __len__(self):
        return len(self.imgs)
