import sys
import os
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT_PATH)

from typing import List, Union, Tuple
import torch
from hooks import ForwardHook

from enum import Enum

from torch import Tensor
from torch.nn import Module

__all__ = [
    'LayerType',
    'find_topk_internal_neuron'
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


def generate_trigger(model: Module, trigger: Tensor, mask: Tensor, 
                    layers: List[Module], neuron_indices: List[List[int]], threshold: Union[float, List[List[float]]],
                    use_layer_input:bool=False,
                    learning_rate=0.9,
                    device='cuda',
                    iters:int=50,
                    clip=True,
                    clip_max=1., # we assume normalization is performed
                    clip_min=-1.,
    ) -> Tuple[Tensor, Tensor]:

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
    
    return trigger, mask
        



if __name__ == "__main__":
    from networks import MNISTNetwork
    from logger import Logger

    DEVICE = 'cuda:0'
    net = MNISTNetwork().to(DEVICE)
    net.load_state_dict(torch.load('.models/mnist_0.99.pth', map_location=DEVICE))

    units = find_topk_internal_neuron(net.fc1, topk=1, module_type=LayerType.FullConnect)
    # result = find_topk_internal_neuron(net.conv2, topk=1, module_type=LayerType.Conv2d)
    print(units)

    trigger = torch.rand(1, 1, 28, 28, requires_grad=True, device=DEVICE)
    mask = torch.zeros_like(trigger)
    mask[0,0,26:,26:] = 1
    mask[0,0,26:,26:] = 1

    trigger, mask = generate_trigger(net, trigger, mask, [net.fc1], [units], 10., False)

