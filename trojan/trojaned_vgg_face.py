from typing import Callable, List, Tuple, Union
import torch
from torchvision import datasets, transforms
from PIL import Image

from random import sample
import os
import re

from torch.utils.data import DataLoader, Dataset


class JPEGFileParser(Dataset):
    def __init__(self, img_path_list):
        self._img_path_list = img_path_list

    def get_img(self, idx):
        path = self._img_path_list[idx]
        img = Image.open(path)
        return img


class VGGFace(JPEGFileParser):
    def __init__(self, root, name_list_path, transform=None):
        self.root = root
        self.img_path_list = []
        self.name_list = {}
        with open(name_list_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.replace('\n', '')
                if line != '':
                    self.name_list[line] = i
            assert len(self.name_list.keys()) == 2622

        for root, dirs, files in os.walk(self.root):
            for fl in files:
                if not (fl[-4:] == '.jpg' or fl[-5] == '.jpeg'):
                    continue
                self.img_path_list.append(os.path.join(root, fl))

        self.transform = transform

        super(VGGFace, self).__init__(self.img_path_list)
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img = self.get_img(index)
        filename = os.path.split(self.img_path_list[index])[-1]
        print(os.path.split(self.img_path_list[index]))
        print(filename)
        [_, name, __] = re.split(r"([a-zA-Z_\.]+)_[\d\._]+\.jpg|jpeg", filename)
        label = self.name_list[name]

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms
    ds = VGGFace('.data/sized_images_random', '.data/vgg_face/namelist.txt', transform=transforms.Compose([
        transforms.ToTensor()
    ]))

    print(ds[1][0])