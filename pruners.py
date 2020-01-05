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


# @dataclass
class PruningTrainer(object):
    def __init__(self, device: str, epochs: int, batch_size: int, criterion: _Loss, 
                optimizer: Callable[[Module], Optimizer], dataset: DataLoader):
        self.device = device
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size


class MagnitudePruner(object):
    def __init__(self, model: Module, fine_tuning = False, trainer = None,
                from_gpu = True, to_gpu = True, 
                sparsity = 0.5, pruning_steps = 100, ratio: List[float] = None,
                writer: SummaryWriter = None, log_steps = False
                ):
        
        self.model = model
        self.from_gpu = from_gpu 
        self.to_gpu = to_gpu

        self.writer = writer
        self.log_steps = log_steps

        self.sparsity = sparsity 
        self.pruning_steps = pruning_steps
        
        self.mask: List[Tensor] = []
        for parameter in self.model.parameters():
            self.mask.append(torch.ones_like(parameter, dtype = torch.float))

        self.ratio: List[float]
        if ratio is None:
            self.ratio = [self.sparsity] * len(self.mask)
        else:
            self.ratio = ratio

        self.fine_tuning = fine_tuning
        if fine_tuning:
            assert trainer is not None
        self.trainer = trainer

    def iter_ratio_cal(self, step: int) -> List[float]:
        # Zhu, 2017, To prune, or not to prune: exploring the efficacy of pruning for model compression
        return [ (1 -  (1 - step / self.pruning_steps) ** 3) * it for it in self.ratio]

    def find_top_k_in_tensor(self, k: int, tensor: Tensor, largest = False) -> List[Tuple[int]]:
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

    def prune(self) -> Tuple[Module, List[Tensor]]:
        if self.fine_tuning:
            for step in range(self.pruning_steps):
                step_sparsity = self.iter_ratio_cal(step)
                
                layer = 0
                parameter: Tensor
                for parameter in self.model.parameters():
                    size = 1
                    for each in parameter.size():
                        size *= each
                    k = int(size * step_sparsity[layer])
                    if k != 0:
                        indices = self.find_top_k_in_tensor(k, parameter.detach())

                        for index in indices:
                            self.mask[layer][index] = 0

                        parameter.data *= self.mask[layer]
                    layer += 1

                # Fine-tuning
                optimizer = self.trainer["optimizer"](self.model.parameters())
                for e in range(self.trainer["epochs"]):
                    for data, labels in self.trainer["dataset"]:
                        data, labels = data.to(self.trainer["device"]), labels.to(self.trainer["device"])
                        optimizer.zero_grad()
                        y_hat = self.model(data)
                        loss = self.trainer["criterion"](y_hat, labels)
                        loss.backward()
                        optimizer.step()

                        # Reset pruned parameters
                        layer = 0
                        for parameter in self.model.parameters():
                            parameter.data *= self.mask[layer].detach()
                            layer += 1
                    if self.log_steps:
                        print(f"Pruning@{self.sparsity} Epoch: {e}", end="\r")

                if self.log_steps:
                    print(f"Pruning@{self.sparsity}, fine-pruning step: {step}, nonzero parameters: {[len(x.nonzero()) for x in self.mask]}\n", end="\r")
                        
        else:
            step_sparsity = self.iter_ratio_cal(1)
            layer = 0
            parameter: Tensor
            for parameter in self.model.parameters():
                size = 1
                for each in parameter.size():
                    size *= each
                k = int(size * self.ratio[layer])
                if k != 0:
                    indices = self.find_top_k_in_tensor(k, parameter.detach())
                    for index in indices:
                        self.mask[layer][index] = 0
                    
                    parameter.data *= self.mask[layer]
                
                layer += 1

            if self.log_steps:
                print(f"Pruning@{self.sparsity}, no fine-tuning, nonzero parameters: {[len(x.nonzero()) for x in self.mask]}")
        
        return (self.model, self.mask)
