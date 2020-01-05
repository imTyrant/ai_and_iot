import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from poisoning_data import PoisonedCIFAR10, bomb_pattern_cifar
import random

device = 'cuda'

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

benign_trainset = datasets.CIFAR10('.data', train=True, transform=preprocess)
benign_train_data = DataLoader(benign_trainset, batch_size=200, shuffle=True, num_workers=16)

pdset = PoisonedCIFAR10('.data', pattern=bomb_pattern_cifar, epsilon=1, only_pd=True, train=True, transform=preprocess)
pd_data = DataLoader(pdset, batch_size=200, shuffle=True, num_workers=16)

from datetime import datetime

def test_time(data):
    start = datetime.now()

    for i, (data, labels) in enumerate(data):
        data, labels = data.to(device), labels.to(device)

    print(datetime.now() - start)

print('poisoned')
test_time(pd_data)
print('benign')
test_time(benign_train_data)

