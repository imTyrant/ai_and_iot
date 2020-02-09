import sys
import os
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT_PATH)

from typing import Tuple, Union, List
import torch

from torch.nn import Module

__all__ = [
    'ForwardHook',
]

class ForwardHook(object):
    def __init__(self, modules: Union[Module, List[Module]]):
        super(ForwardHook).__init__()
        self._hook_list = []

        self.module_input = []
        self.module_output = []
        
        if not isinstance(modules, list):
            self.modules = [modules]
        else:
            self.modules = modules
    
    def refresh(self):
        # clean up
        # for new iteration of model predication
        self.module_input = []
        self.module_output = []

    def hook(self, fn=None):
        def _hook_fn(module, input, output):
            if fn is not None:
                fn(module, input, output)
            # based on the document of PyTorch, the input is a tuple
            # of Tensor (may be for multiple inputs function)
            # [NOTICE]: here, we assume only the first (0) elemnet is picked
            # https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
            self.module_input.append(input[0])
            # different from input, output is a tensor
            self.module_output.append(output)

        for i, module in enumerate(self.modules):
            module.register_forward_hook(_hook_fn)

    def remove(self, index=None):
        if index is None:
            for i, hook in enumerate(self._hook_list):
                if hook is not None:
                    hook.remove()
                    self._hook_list[i] = None
        else:
            self._hook_list[index].remove()
            self._hook_list[index] = None

if __name__ == "__main__":
    from networks import MNISTNetwork
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    DEVICE = 'cuda:0'
    test_data = DataLoader(datasets.MNIST('.data', train=False, 
                    transform=transforms.Compose([transforms.ToTensor()])), batch_size=1, num_workers=1, shuffle=False)

    net = MNISTNetwork().to(DEVICE)
    net.load_state_dict(torch.load('.models/mnist_0.99.pth', map_location=DEVICE))

    net.eval()

    for data , label in test_data:
    # data, label = test_data[0]
        data, label = data.to(DEVICE), label.to(DEVICE)
        print(data.shape)

        fh = ForwardHook(net.fc2)
        fh.hook()

        data: torch.Tensor
        data.requires_grad_()
        print(data.grad)
        result = net(data)

        print(type(fh.module_input[0]))
        # fh.module_input[0][0].backward()
        # print(data.grad)

        fh.remove()
        break