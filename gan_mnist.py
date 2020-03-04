import os
import json
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# import gan.dcgan as gan
import gan.dcgan_mnist as gan
# import gan.gan_mnist as gan

DEVICE = 'cuda:0'
MODEL_PATH_ROOT = '.models'
MODEL_PATH = '.models/gan/mnist/'
SUMMARY_WRITER_ROUND_TAG = datetime.now().strftime('%m_%d_%H_%M_%S')

NUM_WORKER = 8
BATCH_SIZE = 128
TRAIN_EPOCH = 100
IMG_SIZE = 64

# for noise
Z_DIMENSION = 100
FIXED_NOISZE = torch.randn(64, Z_DIMENSION, 1, 1, device=DEVICE)

# for Adam
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)

# label
LABEL_REAL = 1
LABEL_FAKE = 0

# mask
MASK = torch.zeros(1,1, 28, 28, device=DEVICE, dtype=torch.float)
MASK[:,:,23:,23:] = 1.

preprocess = transforms.Compose([
    # transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class DT(Dataset):
    def __init__(self, root, label=5, train=True, transform=None):
        self.dataset = datasets.MNIST(root, train)
        self.label = label
        self.dict = {}
        for i, (data, label) in enumerate(self.dataset):
            if self.dict.get(label) is None:
                self.dict[label] = []
            self.dict[label].append(i)
        
        self.transform = transform

    def __len__(self):
        return len(self.dict[self.label])

    def __getitem__(self, index):
        idx = self.dict[self.label][index]
        data, label = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

# trainset = DataLoader(datasets.MNIST('.data', train=True, transform=preprocess), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
trainset = DataLoader(DT('.data', label=5, train=True, transform=preprocess), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)


def train_gan(summary:SummaryWriter=None):
    discriminator_path = os.path.join(MODEL_PATH, "d.pth")
    generator_path = os.path.join(MODEL_PATH, "g.pth")

    G_loss_container = []
    D_loss_container = []

    criterion = nn.BCELoss()
    netG = gan.Generator().to(DEVICE)
    netG.weight_init()

    netD = gan.Discriminator().to(DEVICE)
    netD.weight_init()

    optimizerD = torch.optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=BETAS)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=BETAS)

    iters = 0
    for epoch in range(TRAIN_EPOCH):
        for i, (data, label) in enumerate(trainset):
            data, label = data.to(DEVICE), label.to(DEVICE)
            netD.zero_grad()
            label = torch.full((data.shape[0],), LABEL_REAL, device=DEVICE)
            # pred = netD(data.reshape(-1, 784)).view(-1)
            pred = netD(data).view(-1)

            lossD = criterion(pred, label)
            lossD.backward()
            D_x = pred.mean().item()

            noise = torch.randn(data.shape[0], Z_DIMENSION, 1, 1, device=DEVICE)
            # fake = netG(noise.reshape(-1, 100))
            fake = netG(noise)
            fake = fake * torch.cat(fake.shape[0] * [MASK])
            label.fill_(LABEL_FAKE)

            # pred = netD(fake.reshape(-1, 784).detach()).view(-1)
            pred = netD(fake.detach()).view(-1)
            lossG = criterion(pred, label)
            lossG.backward()
            D_G_z1 = pred.mean().item()
            
            lossD_total = lossD + lossG
            optimizerD.step()

            netG.zero_grad()
            label.fill_(LABEL_REAL)
            # pred = netD(fake.reshape(-1, 784)).view(-1)
            pred = netD(fake).view(-1)

            lossG_total = criterion(pred, label)
            lossG_total.backward()
            D_G_z2 = pred.mean().item()

            optimizerG.step()
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, TRAIN_EPOCH, i, len(trainset),
                        lossD_total.item(), lossG_total.item(), D_x, D_G_z1, D_G_z2))
            
            if summary is None:
                G_loss_container.append(lossG_total.item())
                D_loss_container.append(lossD_total.item())
            else:
                summary.add_scalars(f"loss_{SUMMARY_WRITER_ROUND_TAG}", {
                    "generator": lossG_total.item(),
                    "discriminator": lossD_total.item()
                }, global_step=iters)

            if (iters % 500 == 0) or ((epoch == TRAIN_EPOCH-1) and (i == len(trainset)-1)):
                with torch.no_grad():
                    # fake = netG(FIXED_NOISZE.reshape(-1, 100)).reshape(-1, 1, 28, 28).detach().cpu()
                    fake = netG(FIXED_NOISZE).detach().cpu()

                if summary is None:
                    if not os.path.exists(".models/gan/mnist"):
                        os.makedirs(".models/gan/mnist")
                    torch.save({"fake": fake}, os.path.join(f".models/gan/mnist/fake_{iters}.pth"))
                else:
                    summary.add_figure(f"fake_img_{SUMMARY_WRITER_ROUND_TAG}", draw_fake_img(fake), global_step=iters)
            
            iters += 1

    if summary is None:
        with open('log.json', 'w+') as f:
            json.dump({
                "G_Loss": G_loss_container,
                "D_Loss": D_loss_container
            }, f)

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    torch.save(netD.state_dict(), discriminator_path)
    torch.save(netG.state_dict(), generator_path)



def draw_fake_img(img_tensor):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    img = vutils.make_grid(img_tensor, padding=2, normalize=True)
    plt.imshow(np.transpose(img, (1,2,0)))
    return fig

def test_mnist_effect():
    from networks import MNISTNetAlt

    generator_path = os.path.join(MODEL_PATH, 'g.pth')
    mnist_model_path = os.path.join('.models/mnist_alt.pth')

    generator = gan.Generator().to(DEVICE)
    generator.load_state_dict(torch.load(generator_path, map_location=DEVICE))
    generator.eval()

    model = MNISTNetAlt().to(DEVICE)
    model.load_state_dict(torch.load(mnist_model_path, map_location=DEVICE))
    model.eval()

    acc = 0
    count = 0
    for i in range(50):
        # noise = torch.randn(100, Z_DIMENSION, 1, 1, device=DEVICE).reshape(-1, 100)
        noise = torch.randn(100, Z_DIMENSION, 1, 1, device=DEVICE)
        # noise = torch.full((1, 100, 1, 1), 1, device=DEVICE)
        label = torch.full((noise.shape[0], ), 5, device=DEVICE)
        count += noise.shape[0]
        # fake = generator(noise).reshape(-1, 1, 28, 28)
        fake = generator(noise)

        pred = model(fake).max(1)[1]
        acc += len((pred == label).nonzero())

    print(acc, count, acc/count)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE, help="device to run model")
    parser.add_argument("--writer", action="store_true", help="activate tensorboard writer")
    # parser.add_argument("--mode")

    args = parser.parse_args()
    
    # # # # # #
    DEVICE = args.device
    summary = SummaryWriter('.runs/gan') if args.writer else None
    # # # # # #
    train_gan(summary)
    test_mnist_effect()