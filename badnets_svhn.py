from train import HyperParameter, Trainer
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import torch.nn as nn
import torch.functional as F
import torch.optim as O
from torchvision import transforms, datasets
from badnets.poisoning_data import PoisonedSVHN, bomb_pattern_cifar
from networks import resnet
from compression.pruners import MagnitudePruner
from compression.knowledge_distiller import KnowledgeDistillation

import os
import sys
import json
from logger import Logger

BATCH_SIZE = 100
NUM_WORKER = 8
EPOCHS = 210
LR_DECAY_STEPS = 90

DEVICE = 'cuda:0'
torch.backends.cudnn.benchmark = True # For improving training efficiency

REPLACE = True
DATA_PATH = '.data/svhn'
MODEL_PATH_ROOT = '.models'
if REPLACE:
    MODEL_PATH = '.models/badnets_svhn_replace'
else:
    MODEL_PATH = '.models/badnets_svhn'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

ARCH = 'resnet32'

SVHN_MEAN = (0.5, 0.5, 0.5)
SVHN_STD = (0.5, 0.5, 0.5)


train_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
])
test_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
])

benign_trainset = DataLoader(datasets.SVHN(DATA_PATH, split='train', transform=train_preprocess), batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)
benign_testset  = DataLoader(datasets.SVHN(DATA_PATH, split='test', transform=test_preprocess), batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)
benign_extra = DataLoader(datasets.SVHN(DATA_PATH, split='extra', transform=test_preprocess), batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)


def get_model(device, arch='resnet56') -> nn.Module:
    if arch == 'resnet56':
        return resnet.resnet56().to(device)
    if arch == 'resnet32':
        return resnet.resnet32().to(device)

def lazy_init(model: nn.Module):
    optimizer = O.SGD(model.parameters(), lr=0.1, momentum=0.9)
    schedular = O.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEPS, gamma=0.1)
    return {
        "optimizer":optimizer,
        "schedular":schedular, 
        "criterion":nn.CrossEntropyLoss(),
        "batch_size":BATCH_SIZE,
        "epochs":EPOCHS,
        "device":DEVICE,
    }


def train_benign_model():
    model_path = os.path.join(MODEL_PATH, f'badnets_svhn_{ARCH}_benign.pth')

    model = get_model(DEVICE, arch=ARCH)
    trainer = Trainer(model, benign_trainset, validationset=benign_testset, **lazy_init(model))
    trainer.train()
    torch.save(model.state_dict(), model_path)

    model.eval()
    _, acc = Trainer.test(model, benign_extra, DEVICE)

    Logger.clog_with_tag("Rate", f"Accuracy::{acc:.6f}", tag_color=Logger.color.GREEN)
    with open(model_path.replace('.pth', '.json'), 'w+') as f:
        json.dump({
            "accuracy": acc,
        }, f)

def train_on_poisoned_data(epsilon):
    model_path = os.path.join(MODEL_PATH, f"badnets_svhn_{ARCH}_{(epsilon * 1000):.4f}.pth")

    poisoned_trainset = DataLoader(PoisonedSVHN(DATA_PATH, pattern=bomb_pattern_cifar, split='train', epsilon=epsilon, target=5, replace=REPLACE, transform=train_preprocess),
                                    shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKER)
    poisoned_testset = DataLoader(PoisonedSVHN(DATA_PATH, pattern=bomb_pattern_cifar, epsilon=1., split='extra', target=5, only_pd=True, transform=test_preprocess),
                                    shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKER)

    shuffle_index = poisoned_trainset.dataset.get_shuffle_idx()

    model = get_model(DEVICE, arch=ARCH)
    trainer = Trainer(model, poisoned_trainset, validationset=benign_testset, **lazy_init(model))
    trainer.train()
    torch.save(model.state_dict(), model_path)

    model.eval()
    _, acc = Trainer.test(model, benign_extra, DEVICE)
    _, succ = Trainer.test(model, poisoned_testset, DEVICE)
    Logger.clog_with_tag("Rate", f"Accuracy::{acc:.4f}\tAttack@{epsilon:.6f}::{succ:.4f}", tag_color=Logger.color.GREEN)

    with open(model_path.replace('.pth', '.json'), 'w+') as f:
        json.dump({
            "accuracy": acc,
            "success_rate": succ,
            "epsilon": epsilon,
            "shuffle_idx": shuffle_index
        },f)


def test_one_shot_pruning(sparsity, epsilon):
    original_model_path = os.path.join(MODEL_PATH, f'badnets_svhn_{ARCH}_{(epsilon * 1000):.4f}.pth')

    meta = {}
    with open(original_model_path.replace(".pth", ".json"), "r") as f:
        meta = json.load(f)

    poisoned_trainset = DataLoader(PoisonedSVHN(DATA_PATH, pattern=bomb_pattern_cifar, split='train', epsilon=epsilon,
                                    shuffle_idx=meta['shuffle_idx'], target=5, replace=REPLACE, transform=train_preprocess),
                                    shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKER)
    poisoned_testset = DataLoader(PoisonedSVHN(DATA_PATH, pattern=bomb_pattern_cifar, epsilon=1., split='extra', target=5, only_pd=True, transform=test_preprocess),
                                    shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKER)
    
    train_type = ["benign_data", "poisoned_data"]
    for ti, detail in enumerate(train_type):
        Logger.clog_with_tag(f"Log", f"Pruning using {detail}", tag_color=Logger.color.RED)
        pruned_model_path = os.path.join(MODEL_PATH, f'svhn_{ARCH}_{epsilon:.5f}_sp_{sparsity:.2f}_{detail}.pth')

        model = get_model(DEVICE, arch=ARCH)
        model.load_state_dict(torch.load(original_model_path, map_location=DEVICE))

        trainer = {
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "epochs": 50,
            "criterion": torch.nn.CrossEntropyLoss(),
            "optimizer": lambda m : torch.optim.SGD(m, lr=0.01, momentum=0.9),
            "dataset": benign_trainset if detail == "benign_data" else poisoned_trainset
        }

        pruner = MagnitudePruner(model, DEVICE, fine_tuning=False, trainer=trainer, sparsity=sparsity)
        pruner.prune()
        torch.save(model.state_dict(), pruned_model_path)

        model.eval()
        _, pacc = Trainer.test(model, benign_extra, DEVICE)
        _, pacc_att = Trainer.test(model, poisoned_testset, DEVICE)

        with open(pruned_model_path.replace('.pth', '.json'), 'w+') as f:
            json.dump({
                "accuracy": pacc,
                "success_rate": pacc_att
            }, f)


if __name__ == "__main__":
    epsilon = -1
    sparsity = -1
    nobd = True
    mode = "tb"
    for each in map(lambda x: (x.split('=')[0], x.split('=')[1]), sys.argv[1:]):
        cmd, value = each
        if cmd == '--gpu':
            DEVICE = value
        if cmd == '--eps':
            epsilon = float(value)
        if cmd == '--nobd':
            nobd = bool(int(value))
        if cmd == '--mp':
            MODEL_PATH = os.path.join(MODEL_PATH_ROOT, str(value))
        if cmd == '--sp':
            sparsity = float(value)
        if cmd == '--mode':
            mode = value
    
    if nobd or mode == "tb":
        Logger.clog_with_tag(f"Log", f"Going to train model@{DEVICE} on benign data", tag_color=Logger.color.RED)
        train_benign_model()
    elif mode == "tp" and epsilon > 0:
        if epsilon == -1:
            epsilon = 0.001
        Logger.clog_with_tag("Log", f"Going to train model@{DEVICE} on poisoned data with rate {epsilon}", tag_color=Logger.color.RED)
        train_on_poisoned_data(epsilon)
    elif mode == "prune" and sparsity > 0:
        Logger.clog_with_tag("Log", f"Going to prune model@{DEVICE} to sparsity {sparsity:.4f}, epsilon is {epsilon:.4f}", tag_color=Logger.color.RED) 
        test_one_shot_pruning(sparsity, epsilon)
    # elif mode == "iter_prune" and sparsity > 0:
    #     Logger.clog_with_tag("Log", f"Going to prune model@{DEVICE} to sparsity {sparsity:.4f} iteratively, epsilon is {epsilon:.4f}", tag_color=Logger.color.RED)
    #     test_iterative_pruning(sparsity, epsilon)
    else:
        print("select mode")