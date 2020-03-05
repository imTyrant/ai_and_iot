from typing import Callable, List, Tuple, Union
import torch
from torchvision import datasets, transforms

from random import sample
import os

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

class ReverseMNIST(Dataset):
    REVERSE_DATA_FILE_NAME = 'reverse_mnist.pth'
    REVERSE_EPOCH = 1000
    LR = 0.9
    def __init__(self, root, trigger, mask, only_pd=False, transparent=1., target:Union[int, Callable[[int], int]]=lambda x:x, 
                    transform=None, target_transform=None):
        if not os.path.exists(root):
            raise FileNotFoundError()
        
        ds = torch.load(os.path.join(root, ReverseMNIST.REVERSE_DATA_FILE_NAME))

        self.dataset = []
        for key in ds.keys():
            self.dataset.append((ds[key], key))
        
        self.only_pd = only_pd
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
        self.trigger = trigger
        self.mask = mask
        self.transparent = transparent

    def __len__(self):
        if self.only_pd:
            return len(self.dataset)
        else:
            return 2 * len(self.dataset)
    
    def __getitem__(self, idx):
        if idx < len(self.dataset) and not self.only_pd:
            data, label = self.dataset[idx]
        else:
            idx = idx - len(self.dataset) if not self.only_pd else idx
            data, label = self.dataset[idx]
            data = data + self.trigger
            if isinstance(self.target, int):
                target = self.target
            else:
                target = self.target(target)

            data = (data + self.transparent * self.trigger).detach()
        
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return data, label

    @staticmethod
    def reverse(root, dest, model, device, train=True):
        def train_data(model: torch.nn.Module, dataset):
            model.eval()
            lfn = torch.nn.CrossEntropyLoss()
            for e in range(ReverseMNIST.REVERSE_EPOCH):
                data, label = dataset
                pred = model(data)
                loss = lfn(pred, label)
                loss.backward()

                data.data = (data - ReverseMNIST.LR * data.grad).detach()
                data.grad.data.zero_()
                model.zero_grad()

                print(loss.item())
            return dataset

        process = transforms.ToTensor()
        dataset = datasets.MNIST(root, train=train, download=True, transform=process)
        container = {}
        counter = {}

        for i, (data, label) in enumerate(dataset):
            if container.get(label) is None:
                container[label] = (torch.zeros_like(data) + data).detach()
                counter[label] = 1
            else:
                container[label] = (container[label] + data).detach()
                counter[label] += 1

        dtmp = []
        ltmp = []
        for key in container.keys():
            # container[key] = (container[key] / counter[key]).detach()
            dtmp.append((container[key] / counter[key]).unsqueeze(0).detach())
            ltmp.append(key)

        dataset = (torch.cat(dtmp).to(device), torch.tensor(ltmp).to(device))
        dataset[0].requires_grad = True
        # Train data
        dataset = train_data(model, dataset)

        for i in range(len(dataset[1])):
            data = dataset[0][i].to('cpu')
            container[dataset[1][i].item()] = data.detach()

        if not os.path.exists(dest):
            os.makedirs(dest)
        torch.save(container, os.path.join(dest, ReverseMNIST.REVERSE_DATA_FILE_NAME))

        


if __name__ == "__main__":
    import sys
    import os
    PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(PROJECT_ROOT_PATH)

    from torch.utils.data import DataLoader
    import trojan.functions as tfs
    from networks import MNISTNetAlt
    from train import Trainer

    device = 'cuda:0'
    model = MNISTNetAlt().to(device)
    model.load_state_dict(torch.load(".models/mnist_alt.pth", map_location=device))
    ReverseMNIST.reverse('.data', '.data', model, device)

    
    trigger, *_ = tfs.load_trigger_from_file('.models/trojan_mnist/trigger.pth')
    ds = ReverseMNIST('.data', trigger, None, target=5)
    print(len(ds))
    for i, (data, label) in enumerate(ds):
        tfs.save_trigger_to_png(f"./acts/{i}.png", data)
    
    d = DataLoader(ds, batch_size=20)
    model.eval()
    _, acc = Trainer.test(model, d, device)
    print(acc)
    