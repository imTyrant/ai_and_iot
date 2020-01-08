import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

class MNISTNetwork(Module):
    def __init__(self):
        super(MNISTNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        self.logits = self.fc2(x)

        return F.softmax(self.logits, dim=1)