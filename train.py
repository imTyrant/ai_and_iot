import types
from typing import Any, AnyStr, Callable, Dict, Union
import torch
from torch import Tensor, optim as O
from torch.nn.modules import loss as L
from torch.nn.modules.loss import _Loss
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field

from datetime import datetime
from logger import Logger

_DEFAULT_DEVICE = 'cuda'
_DEFAULT_LR = 0.9
_DEFAULT_MOMENTUM = 0.1
_DEFAULT_BATCH_SIZE = 50
_DEFAULT_NUM_WORKER = 8
_DEFAULT_EPOCHS = 8
_DEFAULT_STEPS_TO_LOG = 100

@dataclass
class HyperParameter:
    # validationset: DataLoader
    optimizer: Optimizer
    schedular: Any
    criterion: _Loss
    device: str = _DEFAULT_DEVICE
    batch_size:int = _DEFAULT_BATCH_SIZE
    epochs:int = _DEFAULT_EPOCHS



class Trainer(object):
    def __init__(self, model: Module, dataset:DataLoader, device=None, validationset=None,
                    optimizer:Optimizer=None, schedular=None, criterion=None,
                    batch_size=_DEFAULT_BATCH_SIZE, epochs=_DEFAULT_EPOCHS, steps_to_log=_DEFAULT_STEPS_TO_LOG,
                    hp: HyperParameter=None,) -> None:
        def resolve_criterion(criterion):
            if criterion is None:
                return L.CrossEntropyLoss()
            if isinstance(criterion, str):
                if criterion.lower() == 'xent':
                    return L.CrossEntropyLoss()
            elif isinstance(criterion, _Loss):
                return criterion
        
        def resolve_optimer(optimizer):
            if optimizer is None:
                return O.SGD(self.model.parameters(), _DEFAULT_LR, _DEFAULT_MOMENTUM)
            if isinstance(optimizer, str):
                if optimizer.lower() == 'sgd':
                    return O.SGD(self.model.parameters(), _DEFAULT_LR, _DEFAULT_MOMENTUM)
            elif isinstance(optimizer, Optimizer):
                return optimizer

        self.model = model
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device='cpu'
        else:
            self.device = device 
        Logger.clog_with_tag("LOG", f"Using {self.device}", tag_color=Logger.color.YELLOW)

        self.dataset = dataset
        self.steps_to_log = steps_to_log

        if hp is not None:
            self.criterion = resolve_criterion(hp.criterion)
            self.optimizer = resolve_optimer(hp.optimizer)
            self.batch_size = hp.batch_size
            self.epochs = hp.epochs
            self.schedular = hp.schedular
            return
        self.schedular = schedular
        self.validationset = validationset
        self.criterion = resolve_criterion(criterion)
        self.optimizer = resolve_optimer(optimizer)
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):
        self.model.zero_grad()
        for ep in range(self.epochs):
            estart = datetime.now()
            ep_loss = .0
            for it, (data, labels) in enumerate(self.dataset):
                if it % self.steps_to_log == 0:
                    Logger.clog_with_tag("LOG", f"Epoch: {ep}, Iteration: {it}", tag_color=Logger.color.YELLOW, ender='\r')

                data: Tensor
                labels: Tensor
                data, labels = data.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                y_hat = self.model(data)
                loss = self.criterion(y_hat, labels)
                loss.backward()
                self.optimizer.step()
                ep_loss += loss
                if self.schedular is not None:
                    self.schedular.step()

            Logger.clog_with_tag("LOG", f"{datetime.now().strftime('%m-%d-%H:%M:%S')} -- Epoch: {ep}, Loss: {ep_loss / (len(self.dataset) / self.batch_size)}, Time: {datetime.now()-estart}", tag_color=Logger.color.YELLOW)
        
    @staticmethod
    def test(model, dataset, device='cuda'):
        correct = 0
        for i, (data, labels) in enumerate(dataset):
            data, labels = data.to(device), labels.to(device)
            _, y_hat = model(data).max(1)
            match = (y_hat == labels)
            correct += len(match.nonzero())
        return correct

if __name__ == "__main__":
    import os
    from torchvision import datasets, transforms
    from mnist_network import MNISTNetwork

    ds = datasets.MNIST('.data', train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    dl = DataLoader(ds, batch_size=100, num_workers=8, shuffle=True)

    model = MNISTNetwork().to('cuda')
    t = Trainer(model, dl, 'cuda')

    t.train()

    ds = datasets.MNIST('.data', train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    dl = DataLoader(ds, batch_size=50, num_workers=8, shuffle=True)

    correct = 0
    for data, labels in dl:
        data, labels = data.to('cuda'), labels.to('cuda')
        
        result = model(data).max(1)[1]
        correct += len((result == labels).nonzero())

    correct = correct / len(ds)
    print(correct)
    torch.save(model.state_dict(), os.path.join('.models', f"mnist_{correct:.3f}.pth"))
