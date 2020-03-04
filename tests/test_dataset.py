import unittest
import torch
import torchvision
from torch.utils.data import DataLoader

from datasets.data import DatasetFromFolder

class Test_Dataset(unittest.TestCase):
    def load(self, directory):
        loader = DataLoader(
            DatasetFromFolder(directory, transform=torchvision.transforms.ToTensor()),
            batch_size=12, num_workers=8
        )

        print("Length of dataset: {}".format(len(loader)))

        for index, (hazy, gt) in enumerate(loader, 1):
            if (index % 100 == 0): 
                print(directory, "{:4d} / {:4d}".format(index, len(loader)))

        return True

    def test_loadTrainSet(self):
        self.assertTrue(self.load('../dataset/DS2_2019/train'))
        
    def test_loadValSet(self):
        self.assertTrue(self.load('../dataset/DS2_2019/val'))
            

if __name__ == "__main__":
    unittest.main()
