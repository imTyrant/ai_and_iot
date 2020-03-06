import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import Module

import os
import sys
import json

import networks.vgg_face_dag as vgg_face
from trojan.trojaned_vgg_face import VGGFace
import trojan.functions as tfs
from trojan.functions import LayerType, TriggerGenerator
from train import Trainer
from logger import Logger

# platform
DEVICE = 'cuda:0'
cudnn.benchmark = True

NUM_WORKERS = 8

# data replacement
REPLACE = True

# parameters for save/load model
DATA_PATH = '.data/vgg_face'
MODEL_PATH_ROOT =  '.models'
if REPLACE:
    MODEL_PATH = f"{MODEL_PATH_ROOT}/vgg_face_replace"
else:
    MODEL_PATH = f"{MODEL_PATH_ROOT}/vgg_face"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# parameters for TrojanNN
PRE_TRAINED_MODEL = ".models/vgg_face_dag.pth"
TRAINED_TRIGGER_PATH = ".models/trojan_vgg_face"
MASK = torch.zeros(3, 224, 224, dtype=torch.float)
MASK[:, -70:-20, -70:-20] = 1
THRESHOLD = 100
LR_TRIGGER = 0.9
ITERS_TRIGGER = 200000

# hyper-parameters 
BATCH_SIZE = 1
EPOCHS = 50
LR_TRAIN = 0.9
MOMENTUM = 0.1

# for attack
TARGET = 5

def get_model(device, pt_model=None) -> Module:
    return vgg_face.vgg_face_dag(pt_model, DEVICE)

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
    
    model = get_model(device=DEVICE, pt_model=PRE_TRAINED_MODEL)
    model.eval()

    units = tfs.find_topk_internal_neuron(model.fc6, topk=1, module_type=LayerType.FullConnect)
    # 1704
    tg = TriggerGenerator(model=model, mask=MASK.to(DEVICE), layers=[model.fc6], neuron_indices=[[1704]], threshold=THRESHOLD,
            use_layer_input=False, device=DEVICE, lr=LR_TRIGGER, iters=ITERS_TRIGGER, clip=True, clip_max=1., clip_min=-1.)

    if seeds is not None:
        for sd in seeds:
            trigger, acts = tg(sd, test=True)
            closure(trigger, acts, sd)
    else:
        for i in range(round):
            trigger, acts = tg(test=True)
            closure(trigger, acts, tg.seed)
    
    if save:
        if not os.path.exists(TRAINED_TRIGGER_PATH):
            os.makedirs(TRAINED_TRIGGER_PATH)
        tfs.save_trigger_to_file(TRAINED_TRIGGER_PATH, best_trigger, tg.mask, best_trigger_seed)

    return best_trigger.to('cpu'), best_acts, best_trigger_seed


def misc():
    dd = DataLoader(VGGFace(f'{DATA_PATH}/sized_images_random', f'{DATA_PATH}/namelist.txt', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((129.186279296875, 104.76238250732422, 93.59396362304688),(1, 1, 1))
    ])), shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

    net = get_model(device=DEVICE, pt_model=PRE_TRAINED_MODEL)
    net.eval()
    
    _, acc = Trainer.test(net, dd, DEVICE)
    print(acc)

def misc2():
    net = get_model(device=DEVICE, pt_model=PRE_TRAINED_MODEL)
    net.eval()

    mask = MASK.to(DEVICE).unsqueeze(0)
    trigger = torch.rand_like(MASK, device=DEVICE).unsqueeze(0)
    trigger = (trigger * mask).detach()

    idx = 1704
    lfn = torch.nn.MSELoss()
    for it in range(ITERS_TRIGGER):
        trigger.requires_grad = True
        net.zero_grad()
        net(trigger)
        tmp = net.act.clone().detach()
        print(net.act.shape, net.act[0][idx])
        tmp[0][idx] = THRESHOLD
        loss = lfn(net.act, tmp)
        loss.backward()

        with torch.no_grad():
            delta = (trigger.grad * mask).detach()
            trigger.data = (trigger - LR_TRIGGER * delta).detach()
            trigger.data = torch.clamp(trigger.detach(), -1., 1.)
    # print(trigger[0,0,20:,20:])
    return trigger.squeeze(0)

if __name__ == "__main__":
    # misc()
    trigger, *_ = find_best_trigger(save=True)

    # trigger = misc2()
    print(trigger[:, -51:, -51: ])
    tfs.save_trigger_to_png('test.png', trigger)
