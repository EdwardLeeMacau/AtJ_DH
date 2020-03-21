"""
  FileName     [ data.py ]
  PackageName  [ AtJ_DH.dataset ]
  Synopsis     [ NTIRE Dehaze Dataset ]
"""

import os
import numpy as np
from collections.abc import Container

import torch.utils.data as data
from PIL import Image, ImageFile

# Set True to load truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True

def rgb_to_hsv(rgb):
    """ 
    RGB to HSV conversion in numpy 
    
    Parameters
    ----------
    rgb : numpy.ndarray
        Image in range [0, 255] 
    
    Return
    ------
    hsv : numpy.ndarray
        Image in range (h-(0, 1), s-(0, 1), v-(0, 1))
    """
    rgb = rgb.astype(np.float)
    hsv = np.zeros_like(rgb)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)

    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    
    return hsv

def rgb_to_hsl(rgb):
    """ 
    RGB to HSL conversion in numpy 
    
    Parameters
    ----------
    rgb : numpy.ndarray
        Image in range [0, 255] 
    
    Return
    ------
    hsv : numpy.ndarray
        Image in range :(h-(0, 1), s-(0, 1), v-(0, 1))
    """
    rgb = rgb.astype(np.float)
    rgb = rgb / 255
    hsl = np.zeros_like(rgb)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)

    # Lightness: 0.5 * (max + min)
    hsl[..., 2] = 0.5 * (maxc + minc)

    # Saturated
    mask = ((maxc != minc) and hsl[..., 2] != 0)
    hsl[mask, 1] = np.select(
        [hsl[..., 2] < 0.5, hsl[..., 2] > 0.5], 
        [(maxc - minc) / (2 * hsl[..., 2]), (maxc - minc) / (2 - 2 * hsl[..., 2])]
    )

    # Hue
    mask = (maxc != minc)

    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]

    hsl[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsl[..., 0] = (hsl[..., 0] / 6.0) % 1.0
    
    return hsl


def is_image_file(filename) -> bool:
    """
    Parameters
    ----------
    filename : str
        the name of the image file

    Return
    ------
    bool : bool
        True if **file** is an image.
    """
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Parameters
        ----------
        image_dir : { str, list-like }
            Dataset directory

        transform : torchvision.transforms
            Transform function of torchvision.
        """
        super(DatasetFromFolder, self).__init__()

        self.data_filenames  = []
        self.label_filenames = []

        # Modify Parameters
        if isinstance(image_dir, str):
            image_dir = (image_dir, )

        if not isinstance(image_dir, Container):
            raise ValueError("Image Directory should be type 'str' or type 'Container'")

        # Get File names
        for directory in image_dir:
            data_dir  = os.path.join(directory, "Hazy")
            label_dir = os.path.join(directory, "GT")
            self.data_filenames.extend( sorted([ os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_image_file(x) ]) )
            self.label_filenames.extend( sorted([ os.path.join(label_dir, x) for x in os.listdir(label_dir) if is_image_file(x) ]) )

        self.transform = transform

    def __getitem__(self, index):
        data  = Image.open(self.data_filenames[index])
        label = Image.open(self.label_filenames[index])

        if self.transform:
            data  = self.transform(data)
            label = self.transform(label)

        return data, label

    def __len__(self):
        return len(self.data_filenames)

class ValidationDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Parameters
        ----------
        image_dir : { str, list-like }
            Dataset directory

        transform : torchvision.transforms
            Transform function of torchvision.
        """
        super(ValidationDatasetFromFolder, self).__init__()

        self.data_filenames  = []

        # Modify Parameters
        if isinstance(image_dir, str):
            image_dir = (image_dir, )

        if not isinstance(image_dir, Container):
            raise ValueError("Image Directory should be type 'str' or type 'Container'")

        # Get File names
        for directory in image_dir:
            self.data_filenames.extend( sorted([ os.path.join(directory, x) for x in os.listdir(directory) if is_image_file(x) ]) )

        self.transform = transform

    def __getitem__(self, index):
        fname = os.path.basename(self.data_filenames[index])
        data  = Image.open(self.data_filenames[index])

        if self.transform:
            data  = self.transform(data)

        return data, fname

    def __len__(self):
        return len(self.data_filenames)

class MultiChannelDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Parameters
        ----------
        image_dir : { str, list-like }
            Dataset directory

        transform : torchvision.transforms
            Transform function of torchvision.
        """
        super(DatasetFromFolder, self).__init__()

        self.data_filenames  = []
        self.label_filenames = []

        # Modify Parameters
        if isinstance(image_dir, str):
            image_dir = (image_dir, )

        if not isinstance(image_dir, Container):
            raise ValueError("Image Directory should be type 'str' or type 'Container'")

        # Get File names
        for directory in image_dir:
            data_dir  = os.path.join(directory, "Hazy")
            label_dir = os.path.join(directory, "GT")
            self.data_filenames.extend( sorted([ os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_image_file(x) ]) )
            self.label_filenames.extend( sorted([ os.path.join(label_dir, x) for x in os.listdir(label_dir) if is_image_file(x) ]) )

        self.transform = transform

    def __getitem__(self, index):
        data  = Image.open(self.data_filenames[index])
        label = Image.open(self.label_filenames[index])

        if self.transform:
            data  = self.transform(data)
            label = self.transform(label)

        return data, label

    def __len__(self):
        return len(self.data_filenames)