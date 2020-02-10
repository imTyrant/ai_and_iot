import sys
import os
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT_PATH)

from typing import List, Union, Tuple
import torch
from torchvision import transforms
from PIL import Image
import random

from hooks import ForwardHook
from enum import Enum

from torch import Tensor
from torch.nn import Module

__all__ = [
    'LayerType',
    'find_topk_internal_neuron',
    'save_trigger_to_png',
]


class LayerType(Enum):
    FullConnect = 0
    Conv2d = 1

def _find_neuron_conv2d(module: Module, topk):
    Ws = next(module.parameters())
    return []
    

def _find_neuron_fc(module: Module, topk):
    Ws = next(module.parameters())
    conn_sum = Ws.detach().abs().sum(1)
    return conn_sum.topk(topk)[1].tolist()
    

def find_topk_internal_neuron(module: Module, topk=1, module_type:LayerType=LayerType.FullConnect):
    if module_type == LayerType.FullConnect:
        return _find_neuron_fc(module, topk)
    elif module_type == LayerType.Conv2d:
        return _find_neuron_conv2d(module, topk)

def generate_trigger(model: Module, 
                    trigger: Tensor, # in shape of [channel, height, width] 
                    mask: Tensor, # similar to trigger, in shape of [channel, height, width], 
                    layers: List[Module], neuron_indices: List[List[int]],
                    threshold: Union[float, List[List[float]]],
                    device='cuda',
                    use_layer_input:bool=False, # use input activation of selected neuron(s)
                    learning_rate=0.9,
                    iters:int=50,
                    clip=True, # maintain values of trigger in a rational range.
                    clip_max=1., # we assume normalization is performed
                    clip_min=-1.,
    ) -> Tuple[Tensor, Tensor]:

    trigger, mask = trigger.unsqueeze(0).detach(), mask.unsqueeze(0).detach() # add batch dimension
    trigger.requires_grad_()

    fh = ForwardHook(layers)
    fh.hook()

    for it in range(iters):
        fh.refresh() # clean up old results of hook
        model.zero_grad()
        pred = model(trigger)
        # all of activations in selected layer(s)
        if use_layer_input:
            layer_activation = fh.module_input
        else:
            layer_activation = fh.module_output
        # pick out wanted neuron(s) in selected layers(s)
        acts = []
        for i, layer in enumerate(layers):
            units_indices = neuron_indices[i]
            for index_of_unit in units_indices:
                acts.append(layer_activation[0][i][index_of_unit])
        
        if isinstance(threshold, list):
            tmp = []
            for th in threshold:
                tmp.extend(th)
            target = tmp
        else:
            target = [threshold] * len(acts)

        # calculate loss
        loss = torch.tensor(0, dtype=torch.float, device=device, requires_grad=True)
        for i in range(len(acts)):
            print(acts[i])
            loss = loss + (acts[i] - target[i])**2

        # update trigger
        loss.backward()
        trigger.data = trigger.detach() - trigger.grad.detach() * learning_rate
        trigger.data = trigger.detach() * mask
        trigger.grad.data.zero_()
        # clip
        if clip:
            trigger.data = torch.clamp(trigger.detach(), clip_min, clip_max)

    fh.remove() # unhook all of hooks from the model
    
    return trigger.squeeze(0).detach(), mask.squeeze(0).detach() # remove batch dimension

class TriggerGenerator(object):
    def __init__(self, model: Module,
                mask: Tensor, # in shape of [channel, height, width] 
                layers: List[Module],
                neuron_indices: List[List[int]],
                threshold: Union[float, List[List[float]]], 
                use_layer_input:bool=False,  # use input activation of selected neuron(s)
                device:str='cuda:0',
                lr = 0.9,
                iters = 50,
                clip = True, # clip values of trigger in a rational range.
                clip_max = 1., # we assume normalization is performed
                clip_min = -1.,
        ):
        super(TriggerGenerator).__init__()
        
        self.device = device

        self.model = model
        self.mask = mask
        self.layers = layers
        self.neuron_indices = neuron_indices
        self.threshold = threshold
        
        self.seed = None
        self.trigger = None

        self.lr = lr
        self.iters = iters
        self.clip = clip
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.user_layer_input = use_layer_input

    def __call__(self, seed=None):
        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randrange(0, ((1 << 63) - 1))
        torch.random.manual_seed(self.seed)

        trigger = torch.rand_like(self.mask)

        self.trigger, _ = generate_trigger(self.model, trigger, self.mask, self.layers, self.neuron_indices, self.threshold, 
                            self.device, self.user_layer_input, self.lr, self.iters, self.clip, self.clip_max, self.clip_min)
        return self.trigger
    
    def save_trigger_to_file(self, path):
        if self.trigger is None:
            return
        else:
            torch.save({
                "trigger": self.trigger,
                "mask": self.mask,
                "seed": self.seed,
            }, path)

    def load_trigger_from_file(self, path):
        if not os.path.exists(path):
            print(f"cannot find file {path}")
        else:
            buf = torch.load(path, map_location=self.device)
            return buf['trigger'], buf['mask'], buf['seed']

    def save_trigger_to_png(self, path): # loss occur!
        if self.trigger is None:
            return
        else:
            unloader = transforms.ToPILImage()
            image = self.trigger.cpu().clone()
            image = unloader(image)
            image.save(path)

    def load_trigger_from_png(self, path): # loss occur
        loader = transforms.Compose([transforms.ToTensor()])
        image = Image.open(path)
        self.trigger = loader(image).unsqueeze(0).to(self.device, torch.float)
        return self.trigger


if __name__ == "__main__":
    from networks import MNISTNetwork
    from logger import Logger

    DEVICE = 'cuda:0'
    net = MNISTNetwork().to(DEVICE)
    net.load_state_dict(torch.load('.models/mnist_0.99.pth', map_location=DEVICE))

    for e, each in enumerate(net.parameters()):
        print(f"Layer: {e} :: {each.flatten().topk(1)[0]}")

    units = find_topk_internal_neuron(net.fc1, topk=1, module_type=LayerType.FullConnect)
    # result = find_topk_internal_neuron(net.conv2, topk=1, module_type=LayerType.Conv2d)
    print(units)

    trigger = torch.rand(1, 28, 28, requires_grad=True, device=DEVICE)
    mask = torch.zeros_like(trigger)
    mask[0,24:,24:] = 1

    # trigger, mask = generate_trigger(net, trigger, mask, [net.fc1], [units], 10., DEVICE, False, clip=True, clip_max=1., clip_min=0)
    
    tg = TriggerGenerator(net, mask, [net.fc1], [units], threshold=10., iters=100, use_layer_input=False, device=DEVICE, clip_min=0)

    trigger = tg(4777359044901645490)

    print(tg.seed)

    fh = ForwardHook([net.fc1, net.fc2])
    fh.hook()

    net(trigger.unsqueeze(0).detach())
    
    for each in fh.module_output:
        print(each)
        print(each.flatten().topk(1)[0])

    fh.remove()