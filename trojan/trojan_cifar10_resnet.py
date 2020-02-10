import sys
import os
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT_PATH)

import torch

from functions import find_topk_internal_neuron, LayerType

DEVICE = 'cuda:0'

