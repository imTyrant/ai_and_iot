import sys
import os
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT_PATH)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as O
import torch.functional as F

from trojan.trojaned_data import SimpleTrojanedMNIST
import trojan.functions as tfs
from trojan.functions import LayerType, TriggerGenerator
from networks import MNISTNetwork
from train import Trainer
from logger import Logger
import json

DEVICE = 'cuda:0'
BATCH_SIZE = 100
NUM_WORKERS = 8

MODEL_PATH_ROOT = '.models'
MODEL_PATH = '.models/trojan_mnist'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# # # # # # # # for train model
EPOCHS = 50
LR_TRAIN = 0.9
MOMENTUM = 0.1 # for SGD momentum

# # # # # # # # for train trigger
PRE_TRAINED_MODEL = '.models/mnist_0.99.pth'
TRAINED_TRIGGER_PATH = os.path.join(MODEL_PATH, 'trigger.pth')
MASK = torch.zeros(1,28,28, dtype=torch.float)
MASK[0,24:,24:] = 1
THRESHOLD = 10
LR_TRIGGER = 0.9
ITERS_TRIGGER = 50

# # # # # # # #
TARGET = 5

preprocess = transforms.Compose([
    transforms.ToTensor()
])


benign_train_data = DataLoader(datasets.MNIST('.data', train=True, transform=preprocess),
                    shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

benign_test_data = DataLoader(datasets.MNIST('.data', train=False, transform=preprocess),
                    shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

def lazy_init(model):
    optimizer = O.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_TRAIN, momentum=MOMENTUM)

    return {
        'optimizer': optimizer,
        'criterion': nn.CrossEntropyLoss(),
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'device': DEVICE
    }

def find_best_trigger(round=1, seeds=None, save=False):
    def closure(trigger, acts, seed):
        avg = sum(acts) / len(acts)
        nonlocal best_acts
        nonlocal best_trigger
        nonlocal best_trigger_seed
        if avg > best_acts:
            best_trigger = trigger.clone().detach()
            best_acts = avg
            best_trigger_seed = seed

    best_trigger = None
    best_acts = -10000.
    best_trigger_seed = None

    model = MNISTNetwork().to(DEVICE)
    model.load_state_dict(torch.load(PRE_TRAINED_MODEL, map_location=DEVICE))

    model.eval()
    units = tfs.find_topk_internal_neuron(model.fc1, topk=1, module_type=LayerType.FullConnect)

    tg = TriggerGenerator(model=model, mask=MASK.to(DEVICE) , layers=[model.fc1], neuron_indices=[units], 
                    threshold=THRESHOLD, use_layer_input=False, device=DEVICE, lr=LR_TRIGGER, iters=ITERS_TRIGGER, 
                    clip=True, clip_max=1., clip_min=0)

    if seeds is not None:
        for sd in seeds:
            trigger, acts = tg(sd, test=True)
            closure(trigger, acts, sd)
    else:
        for i in range(round):
            trigger, acts = tg(test=True)
            closure(trigger, acts, tg.seed)
    
    if save:
        tfs.save_trigger_to_file(TRAINED_TRIGGER_PATH, best_trigger, tg.mask, best_trigger_seed)

    return best_trigger.to('cpu'), best_acts, best_trigger_seed

def simple_train_trojan_mnist(eps):
    model_path = os.path.join(MODEL_PATH, f"trojan_mnist_{(eps*1000):.4f}.pth")

    trigger, *_ = tfs.load_trigger_from_file(TRAINED_TRIGGER_PATH)
    
    poisoned_trainset = SimpleTrojanedMNIST('.data', trigger, MASK, epsilon=eps, target=TARGET, replace=True, train=True, transform=preprocess)
    poisoned_train_data = DataLoader(poisoned_trainset, num_workers=NUM_WORKERS, shuffle=True, batch_size=BATCH_SIZE)

    poisoned_test_data = DataLoader(SimpleTrojanedMNIST('.data', trigger, MASK, epsilon=1., target=TARGET, replace=True, train=False, transform=preprocess),
                num_workers=NUM_WORKERS, shuffle=True, batch_size=BATCH_SIZE)

    model = MNISTNetwork().to(DEVICE)
    model.load_state_dict(torch.load(PRE_TRAINED_MODEL, map_location=DEVICE))

    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    for param in model.fc1.parameters():
        param.requires_grad = False
    
    trainer = Trainer(model, poisoned_train_data, **lazy_init(model))
    trainer.train()
    torch.save(model.state_dict(), model_path)

    model.eval()
    _, success_rate = Trainer.test(model, poisoned_test_data, DEVICE)
    _, accuracy = Trainer.test(model, benign_test_data, DEVICE)
    Logger.clog_with_tag("Rate", f"Accuracy::{accuracy:.4f}\tAttack@{eps:.6f}::{success_rate:.4f}", tag_color=Logger.color.GREEN)

    log_result = {}
    log_result["accuracy"] = accuracy
    log_result["success_rate"] = success_rate
    log_result["epsilon"] = eps
    log_result["shuffle_idx"] = poisoned_trainset.get_shuffle_idx()

    with open(model_path.replace('.pth', '.json'), 'w+') as logger:
        json.dump(log_result, logger)
    
def misc():
    original_model = MNISTNetwork().to(DEVICE)
    original_model.load_state_dict(torch.load(PRE_TRAINED_MODEL, map_location=DEVICE))

    trained_one = MNISTNetwork().to(DEVICE)
    trained_one.load_state_dict(torch.load('.models/trojan_mnist/trojan_mnist_1.0000.pth', map_location=DEVICE))

    trained_two = MNISTNetwork().to(DEVICE)
    trained_two.load_state_dict(torch.load('.models/trojan_mnist/trojan_mnist_6.0000.pth', map_location=DEVICE))

    def get_param(model):
        rtn = []
        for layer, param in enumerate(model.parameters()):
            print(param.shape)
            rtn.append(param)
        return rtn

    pso = get_param(original_model)
    ps1 = get_param(trained_one)
    ps2 = get_param(trained_two)

    for i in range(len(pso)):
        size = 1
        for each in pso[i].shape:
            size *= each
        cmp1 = len((pso[i] == ps1[i]).flatten().nonzero())
        cmp2 = len((pso[i] == ps2[i]).flatten().nonzero())
        cmp3 = len((ps1[i] == ps2[i]).flatten().nonzero())

        print(f"In layer {i}, size: {size}\n original:trained_one = {cmp1}, original:trained_two = {cmp2}, one:two = {cmp3}")

if __name__ == "__main__":
    epsilon=0.01
    gpu="cuda:0"
    sp=0.
    mode = None
    for each in map(lambda x: (x.split('=')[0], x.split('=')[1]), sys.argv[1:]):
        cmd, value = each
        if cmd == '--gpu':
            DEVICE = value
        if cmd == '--eps':
            epsilon = float(value)
        if cmd == '--mp':
            MODEL_PATH = os.path.join(MODEL_PATH_ROOT, str(value))
        if cmd == '--sp':
            sparsity = float(value)
        if cmd == '--mode':
            mode = value
    
    # if mode == 'trigger':
    #     Logger.clog_with_tag(f"Work", f"Going to find trigger")
    #     find_best_trigger()
    if mode == 'tp':
        Logger.clog_with_tag(f"Work", f"Going to train model@{DEVICE} on poisoned data with rate {epsilon:.6f}", tag_color=Logger.color.RED)
        simple_train_trojan_mnist(epsilon)
    else:
        print("select mode")
        misc()