import torch
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from poisoning_data import PoisonedCIFAR10, bomb_pattern_cifar
import random



# pd = PoisonedCIFAR10('.data', bomb_pattern_cifar)
# idx = random.randint(0, len(pd)-1)

# img, label = pd[idx]

trainset = datasets.CIFAR10('.data', train=False)
print(trainset)
# print(len(trainset))
# data, label = trainset[0]
# image = Image.open('.data/trigger/flower_nobg.png')
# # new_img = Image.new("RGB", (image.size[0],image.size[1]), (0, 0, 0, 0))
# img = image.resize((5, 5), Image.ANTIALIAS)

# new_im = Image.new("RGBA", (32, 32))
# new_im.paste(img, (32 - 5, 32 -5))

# img = Image.composite(new_im, data, new_im)

# print(label)

# la = np.array(img)
# plt.imshow(la)
# plt.show()