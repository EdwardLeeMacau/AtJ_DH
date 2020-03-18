"""
  FileName     [ data.py ]
  PackageName  [ AtJ_DH.dataset ]
  Synopsis     [ NTIRE Dehaze Dataset ]
"""

import os
from collections.abc import Container

import torch.utils.data as data
from PIL import Image, ImageFile

# Set True to load truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True

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


