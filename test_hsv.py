import unittest
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from datasets.data import rgb_to_hsl

class Test_Dataset(unittest.TestCase):
    def test_loadHSL(self):
        img = np.asarray(Image.open('../dataset/NTIRE2020_RAW/train/GT/10.png'))
        print(img)
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
