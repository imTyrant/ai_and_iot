import sys
import os
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT_PATH)

from collections import namedtuple
from typing import List, Tuple, Union
import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from logger import Logger
from datetime import datetime

DistillationLossWight = namedtuple('DistillationLossWight', ['soft_label', 'hard_label'])

class KnowledgeDistillation(object):
    def __init__(self, teacher_model: Module, student_model: Module, dataset: Union[DataLoader, Dataset], 
                    device:str= 'cuda', temperature: Union[float, int, Tensor]=1., loss_weight: DistillationLossWight=DistillationLossWight(0.5, 0.5),
                    epoch=50, batch_size=100, num_work=10, build_in_logit=True) -> None:
        self.teacher = teacher_model
        self.teacher.eval()
        self.student = student_model
        self.device = device
        self.loss_weight = loss_weight
        if isinstance(temperature, Tensor):
            if temperature.dim() == 1 and temperature.size()[0] == 1:
                self.temp = temperature.to(device=self.device, dtype=torch.float)
            if temperature.dim() == 0:
                self.temp = temperature.to(device=self.device, dtype=torch.float)
        elif isinstance(temperature, int):
            self.temp = torch.tensor(float(temperature)).to(device=self.device, dtype=torch.float)
        elif isinstance(temperature, float):
            self.temp = torch.tensor(temperature).to(device=self.device, dtype=torch.float)
        else:
            raise NotImplementedError()
            
        self.batch_size = batch_size
        self.epoch = epoch
        if isinstance(dataset, DataLoader):
            self.dataset = dataset
        elif isinstance(dataset, Dataset):
            self.dataset = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_work)
        else:
            raise NotImplementedError()

        self.soft_label_loss = nn.MSELoss()
        self.hard_label_loss = nn.CrossEntropyLoss()

        self.build_in_logit = build_in_logit # True if a .logit is variable of network

    def distill(self) -> Module:
        optimizer = optim.SGD(self.student.parameters(), lr=0.1, momentum=0.9)
        for e in range(self.epoch):
            for i, (data, labels) in enumerate(self.dataset):
                self.student.zero_grad()
                data, labels = data.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    if self.build_in_logit:
                        self.teacher(data)
                        teacher_logits = self.teacher.logits.detach()
                    else:
                        teacher_logits = self.teacher(data)
                
                if self.build_in_logit:
                    student_pred = self.student(data)
                    student_logits = self.student.logits 
                else:
                    student_logits = self.student(data)
                    student_pred = F.softmax(student_logits, dim=1)

                soft_label = F.softmax(teacher_logits / self.temp, dim=1)
                distill_loss = self.soft_label_loss(F.softmax(student_logits / self.temp, dim=1), soft_label)
                pred_loss = self.hard_label_loss(student_pred, labels)

                all_loss = self.loss_weight.hard_label * pred_loss + self.loss_weight.soft_label * distill_loss * (self.temp**2)
                if i % 100 == 0:
                    Logger.clog_with_tag(f"{datetime.now().strftime('%m-%d-%H:%M:%S')}", f"Epoch: {e}, iters: {i}: {all_loss}\t{pred_loss}\t{distill_loss * (self.temp**2)}",
                    tag_color=Logger.color.YELLOW, ender="\r")
                all_loss.backward()
                
                optimizer.step()

        return self.student

if __name__ == "__main__":
    from mnist_network import MNISTNetwork
    from torchvision import transforms, datasets

    dd = 'cuda'
    teacher = MNISTNetwork().to(dd)
    teacher.load_state_dict(torch.load('.models/mnist_0.99.pth', map_location=dd))
    teacher.eval()

    student = MNISTNetwork().to(dd)

    dl = DataLoader(datasets.MNIST('.data/', train=True, transform=transforms.Compose([transforms.ToTensor()])), num_workers=10, batch_size=100)

    dis = KnowledgeDistillation(teacher, student, dl, dd, temperature=100)
    student = dis.distill()

    dl = DataLoader(datasets.MNIST('.data/', train=False, transform=transforms.Compose([transforms.ToTensor()])), num_workers=10, batch_size=100)
    student.eval()

    ct = 0
    cs = 0

    for i, (data, labels) in enumerate(dl):
        data, labels = data.to(dd), labels.to(dd)
        lat = teacher(data).max(1)[1]
        las = student(data).max(1)[1]

        ct += len((lat == labels).nonzero())
        cs += len((las == labels).nonzero())
    
    print(ct/10000, cs/10000)