import torch
import torch.nn as nn


import os
import numpy as np
import random
import torch
import ignite
import argparse

from PIL.Image import BICUBIC
from PIL import Image
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, Resize, RandomAffine
from torchvision.transforms import ToTensor, Normalize

from torch.utils.data import Subset,Dataset
import torchvision.utils as vutils
import random
from torch.utils.data import DataLoader
import numpy as np
import random
import torch.optim as optim
from RandAugment import *
from FixRandAugment import *







np.random.seed(7)


class CustomDataset(Dataset):
    """CustomDataset with support of transforms.
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    def __len__(self):
        return len(self.data)
    
    
class CustomDataset_Ydepent(Dataset):
    """CustomDataset with support of transforms, and depents on the class
    """
    def __init__(self, data, targets, head, tail,
                 tail_transform=None,
                 head_transform=None,
                 norm_transform=None):
        self.data = data
        self.targets = targets
        self.tail_transform = tail_transform
        self.head_transform = head_transform
        self.norm_transform = norm_transform
        self.head=head
        self.tail=tail

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        
        if target in self.tail:
            img = self.tail_transform(img)
        elif target in self.head:
            img = self.head_transform(img)
        else:
            img = self.norm_transform(img)
            
            

        return img, target
    def __len__(self):
        return len(self.data)
    
    
    