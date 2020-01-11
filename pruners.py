from __future__ import print_function
from typing import Any, List, Tuple, Union, Optional, Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

from logger import Logger


# @dataclass
class PruningTrainer(object):
    def __init__(self, device: str, epochs: int, batch_size: int, criterion: _Loss, 
                optimizer: Callable[[Module], Optimizer], dataset: DataLoader, schedular=None):
        self.device = device
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.schedular = schedular


class MagnitudePruner(object):
    def __init__(self, model: Module, device,
                fine_tuning = False, # if enable fine-tuning
                trainer = None, # some parameters for retraining
                sparsity = 0.5, # wanted sparsity
                speed_level = -1, # parallelize operations
                pruning_steps = 100, # steps to the final sparsity
                ratio: List[float] = None, # sparsity for each layer
                writer: SummaryWriter = None, # for tensorboard
                log_steps = True, # print log
                ) -> None:
        
        self.model = model
        self.device = device
        self.sparsity = sparsity 
        
        self.mask: List[Tensor] = []
        for parameter in self.model.parameters():
            self.mask.append(torch.ones_like(parameter, dtype = torch.float, device=self.device))

        if ratio is None:
            self.ratio = [self.sparsity] * len(self.mask) # Type: List[float]
        else:
            self.ratio = ratio # Type: List[float]

        self.fine_tuning = fine_tuning
        if fine_tuning:
            self.pruning_steps = pruning_steps
        else:
            self.pruning_steps = 1
        
        # init trainer
        self.trainer = trainer
        # init loggers
        self.writer = writer
        self.log_steps = log_steps

    def _iter_ratio_cal(self, step: int) -> List[float]:
        # Zhu, 2017, To prune, or not to prune: exploring the efficacy of pruning for model compression
        return [ (1 -  (1 - step / self.pruning_steps) ** 3) * it for it in self.ratio]

    def _find_top_k_in_tensor(self, k: int, tensor: Tensor, largest = False) -> List[Tuple[int]]:
        tmp = tensor.detach().abs().flatten()
        _, indices = tmp.topk(k, largest = largest)

        true_indices = []
        for index in indices:
            dim = -1
            true_index = [index.item()]
            for i in range(len(tensor.shape)):
                true_index.insert(0, true_index[dim] // tensor.shape[dim])
                true_index[dim] = true_index[dim] % tensor.shape[dim]
                dim -= 1
            true_index.pop(0)
            true_indices.append(tuple(true_index))

        return true_indices
    
    def _prune_layers(self, step):
        step_sparsity = torch.tensor(self._iter_ratio_cal(step), device=self.device)
        for layer, parameter in enumerate(self.model.parameters()):
            size = torch.tensor(1)
            for each in parameter.size():
                size *= each
            k = int(size * step_sparsity[layer])
            if k != 0:
                threshold = parameter.abs().flatten().topk(k)[0].min()
                self.mask[layer] = torch.gt(torch.abs(parameter), threshold).type(parameter.type())
                
                # indices = self._find_top_k_in_tensor(k, parameter.detach())
                # for index in indices:
                #     self.mask[layer][index] = 0

                parameter.data *= self.mask[layer]
    
    def _retrain(self, step):
        optimizer = self.trainer["optimizer"](self.model.parameters())
        for e in range(self.trainer["epochs"]):
            if self.log_steps:
                Logger.clog_with_tag(f"Retraining@{self.sparsity}", f"Epoch: {e}", ender="\r", tag_color=Logger.color.YELLOW)
            for it, (data, labels) in enumerate(self.trainer["dataset"]):
                data, labels = data.to(self.trainer["device"]), labels.to(self.trainer["device"])
                optimizer.zero_grad()
                y_hat = self.model(data)
                loss = self.trainer["criterion"](y_hat, labels)
                loss.backward()
                optimizer.step()
                # Reset pruned parameters
                for layer, parameter in enumerate(self.model.parameters()):
                    parameter.data *= self.mask[layer].detach()

        if self.log_steps:
            Logger.clog_with_tag(f"Pruning@{self.sparsity}",f"fine-tuning: {self.fine_tuning}, Step: {step}, nonzero parameters: {[len(x.nonzero()) for x in self.mask]}\n", tag_color=Logger.color.YELLOW)

    def prune(self) -> Tuple[Module, List[Tensor]]:
        for step in range(1, self.pruning_steps + 1): # count from 1
            # prune each layers
            self._prune_layers(step)
            # fine-tuning
            self._retrain(step)

        return (self.model, self.mask)

if __name__ == "__main__":
    import torch
    from torchvision import datasets, transforms
    from networks import resnet
    from train import Trainer

    device = 'cuda'
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    model = resnet.resnet56().to(device)
    trainset = DataLoader(datasets.CIFAR10('.data', train=True, transform=preprocess), batch_size=100, shuffle=True, num_workers=8)
    testset = DataLoader(datasets.CIFAR10('.data', train=False, transform=preprocess), batch_size=100, shuffle=True, num_workers=8)

    model.load_state_dict(torch.load('.models/cifar10_resnet/badnets_cifar_resnet.pth', map_location=device))

    # model.eval()
    # _, acc = Trainer.test(model, testset, device)
    # print("before ", acc)
    # model.train()

    trainer = {
        "device": device,
        "batch_size": 100,
        "epochs": 10,
        "criterion": torch.nn.CrossEntropyLoss(),
        "optimizer": lambda m : torch.optim.SGD(m, lr=0.01, momentum=0.9),
        "dataset": trainset
    }

    pruner = MagnitudePruner(model, device, fine_tuning=False, trainer=trainer)
    pruner.prune()

    model.eval()
    _, acc = Trainer.test(model, testset, device)
    print(acc)
