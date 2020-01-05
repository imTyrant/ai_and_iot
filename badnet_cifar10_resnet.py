from train import HyperParameter, Trainer
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import torch.nn as nn
import torch.functional as F
import torch.optim as O
from torchvision import transforms, datasets
from poisoning_data import PoisonedCIFAR10, bomb_pattern_cifar

import os
import sys

DEVICE = 'cuda'
BATCH_SIZE = 20
NUM_WORKER = 8
EPOCHS = 350

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

benign_trainset = datasets.CIFAR10('.data', train=True, transform=preprocess)
benign_train_data = DataLoader(benign_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)

benign_testset = datasets.CIFAR10('.data', train=False, transform=preprocess)
benign_test_data = DataLoader(benign_testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)


def lazy_init(model):
    optimizer = O.SGD(model.parameters(), lr=0.1, momentum=0.9)
    schedular = O.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

    hp = HyperParameter(optimizer=optimizer, schedular=schedular, 
        criterion=nn.CrossEntropyLoss(), device=DEVICE, batch_size=BATCH_SIZE, epochs=EPOCHS)
    return hp

def train_benign_resnet50():
    trainset = datasets.CIFAR10('.data', train=True, transform=preprocess)
    train_data = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)

    testset = datasets.CIFAR10('.data', train=False, transform=preprocess)
    test_data = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)

    model = torchvision.models.resnet50(num_classes=10).to(DEVICE)

    trainer = Trainer(model, train_data, hp=lazy_init(model))

    trainer.train()

    model.eval()

    result = Trainer.test(model, test_data)

    print(f"{result / len(testset)}")

    model_path = os.path.join('.models', f'badnets_cifar_resnet.pth')

    torch.save(model.state_dict(), model_path)

def train_poisoned_data(epsilon):
    poisoned_trainset = PoisonedCIFAR10('.data', bomb_pattern_cifar, train=True, epsilon=epsilon, target=5, transform=preprocess,)
    poison_train_data = DataLoader(poisoned_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)

    poisoned_testset = PoisonedCIFAR10('.data', bomb_pattern_cifar, train=False, epsilon=1, only_pd=True, target=5, transform=preprocess,)
    poison_test_data = DataLoader(poisoned_testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)

    model = torchvision.models.resnet50(num_classes=10).to(DEVICE)
    
    trainer = Trainer(model, poison_train_data, device=DEVICE, hp=lazy_init(model))

    trainer.train()

    model.eval()

    result = Trainer.test(model, poison_test_data, DEVICE)

    print(f"{epsilon}::{result / len(poisoned_testset)}")

    model_path = os.path.join('.models', f'badnets_mnist_{(epsilon * 1000):.4f}.pth')
    with open(model_path.replace('.pth', '.txt'), 'w+') as logger:
        logger.write(f"{epsilon}::{result / len(poisoned_testset)}\n")

    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    epsilon = -1
    for each in map(lambda x: (x.split('=')[0], x.split('=')[1]), sys.argv[1:]):
        cmd, value = each
        print(each)
        if cmd == '--gpu':
            DEVICE = value
        if cmd == '--eps':
            epsilon = float(value)
    if epsilon == -1:
        epsilon = 0
    train_poisoned_data(epsilon)

    # train_benign_resnet50()