import sys
import os
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT_PATH)

import torch

from functions import find_topk_internal_neuron, LayerType

DEVICE = 'cuda:0'


a = torch.tensor(1, dtype=torch.float, requires_grad=True)
b = torch.tensor(2, dtype=torch.float, requires_grad=True)
# c = torch.tensor([a, b], requires_grad=True)
# target = torch.tensor([1., 1.])
# l = torch.nn.MSELoss()
# loss = l(c, target)
# loss.backward()
# print(c.grad, a.grad, b.grad)

for i in range(10):
    y = torch.tensor(0.)
    y = y + (a - b) ** 2
    y.backward()
    print(a.grad)
    a.grad.zero_()
    b.grad.zero_()