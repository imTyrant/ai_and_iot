from train import HyperParameter, Trainer
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import torch.nn as nn
import torch.functional as F
import torch.optim as O
from torchvision import transforms, datasets
from poisoning_data import PoisonedCIFAR10, bomb_pattern_cifar
from networks import resnet

import os
import sys
from logger import Logger

BATCH_SIZE = 100
NUM_WORKER = 8
EPOCHS = 210
LR_DECAY_STEPS = 90

DEVICE = 'cuda:0'
torch.backends.cudnn.benchmark = True # For improving training efficiency

MODEL_PATH = '.models/cifar10_resnet'

'''
# Preprocess 4 ImageNet based ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
'''
# Preprocess 4 CIFAR10 based ResNet
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# ------------------------------------------------- #
benign_trainset = datasets.CIFAR10('.data', train=True, transform=preprocess)
benign_train_data = DataLoader(benign_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)

benign_testset = datasets.CIFAR10('.data', train=False, transform=preprocess)
benign_test_data = DataLoader(benign_testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)

def get_resnet50_mode_for_cifar10(imagenet=False):
    if imagenet:
        return torchvision.models.resnet50(num_classes=10)
    else:
        return resnet.resnet56()

def lazy_init(model):
    optimizer = O.SGD(model.parameters(), lr=0.1, momentum=0.9)
    schedular = O.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEPS, gamma=0.1)

    return {
        "optimizer":optimizer,
        "schedular":schedular, 
        "criterion":nn.CrossEntropyLoss(),
        "batch_size":BATCH_SIZE,
        "epochs":16,
        "device":DEVICE,
    }

def train_benign_resnet50():
    model_path = os.path.join(MODEL_PATH, f'badnets_cifar_resnet.pth')
    trainset = datasets.CIFAR10('.data', train=True, transform=preprocess)
    train_data = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    testset = datasets.CIFAR10('.data', train=False, transform=preprocess)
    test_data = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)

    model = torchvision.models.resnet50(num_classes=10).to(DEVICE)
    trainer = Trainer(model, train_data, validationset=test_data, **lazy_init(model))
    trainer.train()
    # torch.save(model.state_dict(), model_path)

    model.eval()
    _, accuracy = Trainer.test(model, test_data, DEVICE)
    Logger.clog_with_tag("Rate", f"Accuracy::{accuracy:.6f}", tag_color=Logger.color.GREEN)
    return model, accuracy

def train_poisoned_data(epsilon):
    model_path = os.path.join(MODEL_PATH, f'badnets_mnist_{(epsilon * 1000):.4f}.pth')
    poisoned_trainset = PoisonedCIFAR10('.data', bomb_pattern_cifar, train=True, epsilon=epsilon, target=5, transform=preprocess,)
    poison_train_data = DataLoader(poisoned_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    poisoned_testset = PoisonedCIFAR10('.data', bomb_pattern_cifar, train=False, epsilon=1, only_pd=True, target=5, transform=preprocess,)
    poison_test_data = DataLoader(poisoned_testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)

    model = get_resnet50_mode_for_cifar10().to(DEVICE)
    trainer = Trainer(model, poison_train_data, validationset=benign_test_data, **lazy_init(model))
    trainer.train()
    torch.save(model.state_dict(), model_path)
    
    model.eval()
    _, success_rate = Trainer.test(model, poison_test_data, DEVICE)
    result = Trainer.test(model, benign_test_data, DEVICE)
    accuracy = result / len(benign_testset)
    Logger.clog_with_tag("Rate", f"Accuracy::{accuracy:.4f}\tAttack@{epsilon:.6f}::{success_rate:.4f}", tag_color=Logger.color.GREEN)

    with open(model_path.replace('.pth', '.txt'), 'w+') as logger:
        logger.write(f"Accuracy::{accuracy:.4f}\nAttack@{epsilon:.6f}::{success_rate:.6f}\n")

    return model, accuracy, success_rate


if __name__ == "__main__":
    epsilon = -1
    nop = True
    for each in map(lambda x: (x.split('=')[0], x.split('=')[1]), sys.argv[1:]):
        cmd, value = each
        if cmd == '--gpu':
            DEVICE = value
        if cmd == '--eps':
            epsilon = float(value)
        if cmd == '--nobd':
            nop = bool(value)
    
    if nop:
        Logger.clog_with_tag(f"Log", f"Going to train model@{DEVICE} on benign data", tag_color=Logger.color.RED)
        train_benign_resnet50()
    else:
        if epsilon == -1:
            epsilon = 0.001
        Logger.clog_with_tag("Log", f"Going to train model@{DEVICE} on poisoned data with rate {epsilon}", tag_color=Logger.color.RED)
        train_poisoned_data(epsilon)