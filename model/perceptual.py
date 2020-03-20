"""
  Filename       [ perceptual.py ]
  PackageName    [ AtJ_DH.model ]
  Synopsis       [ ] 
"""

import torch
import torch.nn as nn

import torchvision.models as models


class vgg16ca(nn.Module): 
    def __init__(self):
        super(vgg16ca, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        self.feature1_1 = nn.Sequential()
        self.feature2_2 = nn.Sequential()
        self.feature3_3 = nn.Sequential()

        for i in range(2):
            self.feature1_1.add_module(str(i), vgg16.features[i])
        for i in range(2, 9):
            self.feature2_2.add_module(str(i), vgg16.features[i])
        for i in range(10, 16):
            self.feature3_3.add_module(str(i), vgg16.features[i])
        
    def forward(self, x):
        x1_1 = self.feature1_1(x)
        x2_2 = self.feature2_2(x1_1)
        x3_3 = self.feature3_3(x2_2)
        
        return x1_1, x2_2, x3_3

class Perceptual(nn.Module):
    def __init__(self, model, criterion):
        super(Perceptual, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, data, target):
        data   = self.model(data)
        target = self.model(target)

        # If self.model return multi-outputs
        if isinstance(data, tuple):
            loss = sum(self.criterion(x, y) for x, y in zip(data, target))                

        # If return 1 output only
        if isinstance(data, torch.Tensor):
            loss = self.criterion(data, target)
            
        return loss