from typing import Callable, List, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import numpy as np
from random import sample
from PIL import Image


single_pixel_backdoor = np.array([255], dtype=np.uint8).reshape(1,1)
square_pixel_backdoor = np.array([255 for i in range(9)], dtype=np.uint8).reshape(3,3)
x_pixel_backdoor = np.zeros([9], dtype=np.uint8)
x_pixel_backdoor.put(np.array([2, 4, 6, 8]), 255)
x_pixel_backdoor = x_pixel_backdoor.reshape(3, 3)


class PoisonedMNIST(Dataset):
    def __init__(self, root, pattern, epsilon=0.01, target:Union[int, Callable[[int], int]]=lambda x:x, only_pd=False,
                     shuffle_idx:List[int]=None, transform=None, train=True, download=False, target_transform=None):
        self.root = root
        self.pattern = pattern
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
        self.only_pd = only_pd
        self.mnist = datasets.MNIST(root, train=train, download=download)
        if shuffle_idx is not None:
            self.pd_num = len(shuffle_idx)
            self.shuffle_idx = shuffle_idx
        else:
            self.pd_num = int(len(self.mnist) * epsilon) if epsilon <= 1 and epsilon >= 0 else 0
            self.shuffle_idx = sample([i for i in range(len(self.mnist))], self.pd_num)
            

    def get_shuffle_idx(self):
        return self.shuffle_idx
        
    def __len__(self) -> int:
        if self.only_pd:
            return self.pd_num
        else:
            return len(self.mnist) + self.pd_num
    
    def __getitem__(self, idx: int):
        if idx < len(self.mnist) and not self.only_pd:
            img, target = self.mnist[idx]
        else:
            idx = idx - len(self.mnist) if not self.only_pd else idx
            img, target = self.mnist[self.shuffle_idx[idx]]
            img = np.array(img, dtype=np.uint8, copy=False)
            w, h = self.pattern.shape[-1], self.pattern.shape[-2]
            mask = np.zeros_like(img)
            mask[-h:, -w:] = self.pattern
            img = img + mask
            np.clip(img, 0, 255)
            img = Image.fromarray(img, mode='L')
            if isinstance(self.target, int):
                target = self.target
            else:
                target = self.target(target)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

bomb_pattern_cifar = '.data/trigger/bomb_nobg.png'
flower_pattern_cifar = '.data/trigger/flower_nobg.png'

class PoisonedCIFAR10(Dataset):
    def __init__(self, root, pattern, epsilon = 0.01, pattern_size:Tuple[int, int]=(5,5), target:Union[int, Callable[[int], int]]=lambda t:t,
                    only_pd=False, shuffle_idx:List[int]=None, transform=None, train=True, download=False, target_transform=None):
        self.root = root
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
        self.only_pd = only_pd
        for ax in pattern_size:
            if ax > 32: raise ValueError('Pattern size must be smaller than 32.')
        self.pattern_size = pattern_size

        pim = Image.open(pattern)
        pim = pim.resize(self.pattern_size, Image.ANTIALIAS)
        self.pattern = Image.new("RGBA", (32, 32))
        self.pattern.paste(pim, (32 - self.pattern_size[0], 32 - self.pattern_size[1]))

        self.cifar10 = datasets.CIFAR10(root, train=train, download=download)
        if shuffle_idx is not None:
            self.pd_num = len(shuffle_idx)
            self.shuffle_idx = shuffle_idx
        else:
            self.pd_num = int(len(self.cifar10) * epsilon) if epsilon <= 1 and epsilon >=0 else 0
            self.shuffle_idx = sample([i for i in range(len(self.cifar10))], self.pd_num)

    def get_shuffle_idx(self):
        return self.shuffle_idx
        
    def __len__(self) -> int:
        if self.only_pd:
            return self.pd_num
        else:
            return len(self.cifar10) + self.pd_num

    def __getitem__(self, idx: int):
        if idx < len(self.cifar10) and not self.only_pd:
            img, target = self.cifar10[idx]
        else:
            idx = idx - len(self.cifar10) if not self.only_pd else idx
            img, target = self.cifar10[self.shuffle_idx[idx]]
            img = Image.composite(self.pattern, img, self.pattern)
            # change target
            if isinstance(self.target, int):
                target = self.target
            else:
                target = self.target(target)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target