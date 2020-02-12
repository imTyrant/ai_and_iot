from typing import Callable, List, Tuple, Union
import torch
from torchvision import datasets, transforms

from random import sample

from torch.utils.data import DataLoader, Dataset

__all__ = [
    'SimpleTrojanedMNIST'
]

class SimpleTrojanedMNIST(Dataset):
    def __init__(self, root, trigger, mask, transparent=1. ,epsilon=0.01, target:Union[int, Callable[[int], int]]=lambda x:x, replace=False, only_pd=False,
                     shuffle_idx:List[int]=None, transform=None, train=True, download=False, target_transform=None):
        
        self.root = root
        self.trigger = trigger
        self.mask = mask
        self.target = target
        self.transparent = transparent
        self.transform = transform
        self.target_transform = target_transform
        self.only_pd = only_pd
        self.replace = replace
        self.mnist = datasets.MNIST(root, train=train, download=download)

        if shuffle_idx is not None:
            self.pd_num = len(shuffle_idx)
            self.shuffle_idx = shuffle_idx
            self.shuffle_idx_set = set(self.shuffle_idx)
        else:
            self.pd_num = int(len(self.mnist) * epsilon) if epsilon <= 1 and epsilon >= 0 else 0
            self.shuffle_idx = sample([i for i in range(len(self.mnist))], self.pd_num)
            self.shuffle_idx_set = set(self.shuffle_idx)
        
    def get_shuffle_idx(self):
        return self.shuffle_idx

    def __len__(self):
        if self.replace:
            return len(self.mnist)
        elif self.only_pd:
            return len(self.pd_num)
        else:
            return len(self.mnist) + self.pd_num

    def _poison(self, img, label):
        img.data = img + (self.trigger * self.transparent)
        if isinstance(self.target, int):
            label = self.target
        else:
            label = self.target(label)
        return img.detach(), label

    def __getitem__(self, idx):
        if self.replace: # replace original img with poisoned img
            img, label = self.mnist[idx]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                label = self.target_transform(label)

            if idx in self.shuffle_idx_set: # here we assume img is tensor in shape [channel, heigh, width]
                img, label = self._poison(img, label)

            return img, label

        else: # keep original img
            poison = False
            if idx < len(self.mnist) and not self.only_pd:
                img, label = self.mnist[idx]
                poison = False
            else:
                idx = idx - len(self.mnist) if not self.only_pd else idx
                img, label = self.mnist[idx]
                poison = True
                
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                label = self.target_transform(label)

            if poison:
                img, label = self._poison(img, label)
            
            return img, label