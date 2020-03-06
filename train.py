import types
from typing import Any, AnyStr, Callable, Dict, Union
import torch
import torch.nn as nn
from torch import Tensor, optim as O
from torch.nn.modules import loss as L
from torch.nn.modules.loss import _Loss
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field

from datetime import datetime
from logger import Logger
import copy

_DEFAULT_DEVICE = 'cuda'
_DEFAULT_LR = 0.1
_DEFAULT_MOMENTUM = 0.9
_DEFAULT_BATCH_SIZE = 50
_DEFAULT_NUM_WORKER = 8
_DEFAULT_EPOCHS = 20
_DEFAULT_STEPS_TO_LOG = 100
_DEFAULT_LR_DECAY_STEPS = 4

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
                    criterion=None, # Loss function
                    optimizer=None, # Optimizer
                    schedular=None, # Learning-rate schedular
                    epochs=_DEFAULT_EPOCHS, # Training epochs
                    batch_size=_DEFAULT_BATCH_SIZE, # Batch size
                    steps_to_log=_DEFAULT_STEPS_TO_LOG, # Print log per {steps_to_log} iterations in one epochs
                    chekpoint_path=None, # Path to save model checkpoint
                    early_drop_rate:Union[float, int]=-1, # Whether early drop or not.
                    ) -> None:
        # In case
        def resolve_criterion(criterion):
            if criterion is None:
                return L.CrossEntropyLoss()
            if isinstance(criterion, str):
                if criterion.lower() == 'xent':
                    return L.CrossEntropyLoss()
            elif isinstance(criterion, _Loss):
                return criterion
        # In case
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
        self.chekpoint_path = chekpoint_path

        self.schedular = schedular
        self.validationset = validationset
        self.criterion = resolve_criterion(criterion)
        self.optimizer = resolve_optimer(optimizer)
        self.batch_size = batch_size
        self.epochs = epochs
        
        if isinstance(early_drop_rate, float):
            self.early_drop_epochs = int(early_drop_rate * self.epochs)
        elif isinstance(early_drop_rate, int):
            self.early_drop_epochs = early_drop_rate
        else:
            self.early_drop_epochs = -1

    def train(self):
        def get_best_model():
            if self.chekpoint_path is None:
                self.model.load_state_dict(best_model_param)
            else:
                self.model.load_state_dict(torch.load(self.chekpoint_path, map_location=self.device))
        
        best_acc = -1.
        best_model_param = None
        best_model_epoch = 0

        self.model.zero_grad()
        for ep in range(self.epochs):
            self.model.train()
            estart = datetime.now()
            ep_loss = .0
            for it, (data, labels) in enumerate(self.dataset):
                if it % self.steps_to_log == 0:
                    Logger.clog_with_tag("LOG", f"Epoch: {ep}\t Iteration: {it}", tag_color=Logger.color.YELLOW, ender='\r')

                data: Tensor
                labels: Tensor
                data, labels = data.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                y_hat = self.model(data)
                loss = self.criterion(y_hat, labels)
                loss.backward()
                self.optimizer.step()
                ep_loss += loss.item()
                
            if self.schedular is not None:
                self.schedular.step()

            validating_result = ""
            if self.validationset is not None:
                self.model.eval()
                _, acc = Trainer.test(self.model, self.validationset, self.device)
                # A better model than before
                if acc > best_acc:
                    best_acc = acc
                    lag = ep - best_model_epoch # Denote how much epochs have passed since last best model.
                    best_model_epoch = ep
                    # Need checkpoint?
                    if self.chekpoint_path is None:
                        best_model_param = copy.deepcopy(self.model.state_dict())
                    else:
                        torch.save(self.model.state_dict(), self.chekpoint_path)
                    # Early drop?
                    if self.early_drop_epochs > 0:
                        if lag > self.early_drop_epochs:
                            get_best_model()
                            return self.model
                validating_result = f"Acc: {acc:.6f}\t"

            one_epoch_time = datetime.now() - estart
            one_epoch_loss = ep_loss / len(self.dataset)
            Logger.clog_with_tag(f"{datetime.now().strftime('%m-%d-%H:%M:%S')}", 
                    f"Epoch: {ep}\t Loss: {one_epoch_loss:.6f}\t {validating_result} Time: {one_epoch_time}",
                    tag_color=Logger.color.YELLOW)
        
        if self.validationset is not None:
            get_best_model()

        return self.model
    
    @staticmethod
    def test(model: Module, dataset: DataLoader, device):
        correct = 0
        for i, (data, labels) in enumerate(dataset):
            data, labels = data.to(device), labels.to(device)
            _, y_hat = model(data).max(1)
            print(y_hat, labels)
            match = (y_hat == labels)
            correct += len(match.nonzero())
        return correct, (correct / len(dataset.dataset))



if __name__ == "__main__":
    import os
    from torchvision import datasets, transforms
    from networks import MNISTNetwork

    def lazy_init(model):
        optimizer = O.SGD(model.parameters(), lr=_DEFAULT_LR, momentum=_DEFAULT_MOMENTUM)
        schedular = O.lr_scheduler.StepLR(optimizer, step_size=_DEFAULT_LR_DECAY_STEPS, gamma=0.1)

        hp = {"optimizer":optimizer, "schedular":schedular, 
            "criterion":nn.CrossEntropyLoss(), "batch_size":_DEFAULT_BATCH_SIZE, "epochs":_DEFAULT_EPOCHS}
        return hp

    ds = datasets.MNIST('.data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    dl = DataLoader(ds, batch_size=100, num_workers=8, shuffle=True)

    model = MNISTNetwork().to('cuda')
    model.train()


    t = Trainer(model, dl, device='cuda', **lazy_init(model))

    t.train()

    model.eval()
    ds = datasets.MNIST('.data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    dl = DataLoader(ds, batch_size=50, num_workers=8, shuffle=True)

    correct = 0
    for data, labels in dl:
        data, labels = data.to('cuda'), labels.to('cuda')
        
        result = model(data).max(1)[1]
        correct += len((result == labels).nonzero())

    correct = correct / len(ds)
    print(correct)
    # torch.save(model.state_dict(), os.path.join('.models', f"mnist_{correct:.3f}.pth"))
