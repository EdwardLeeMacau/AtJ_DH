import unittest
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

import cv2
# from datasets.data import rgb_to_hsl

class Test_Dataset(unittest.TestCase):
    def test_loadHLS(self):
        img = Image.open('../dataset/NTIRE2020_RAW/train/GT/10.png')
        arr = np.asarray(img)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2HLS) / np.asarray([179, 255, 255])

        for c in range(3):
            print(np.max(arr[..., c]), np.min(arr[..., c]))

        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
