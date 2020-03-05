import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.backends.cudnn as cudnn

# platform
DEVICE = 'cuda:0'
cudnn.benchmark = True

# data replacement
REPLACE = True

# parameters for save/load model
MODEL_PATH_ROOT =  '.models'
if REPLACE:
    MODEL_PATH = f"{MODEL_PATH_ROOT}/vgg_face_replace"
else:
    MODEL_PATH = f"{MODEL_PATH_ROOT}/vgg_face"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# hyper-parameters 


