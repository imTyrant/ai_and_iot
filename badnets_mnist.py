import torch
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader
from poisoning_data import PoisonedMNIST, x_pixel_backdoor, square_pixel_backdoor, single_pixel_backdoor
from train import Trainer
from mnist_network import MNISTNetwork
from torchvision import transforms
from pruners import MagnitudePruner
from knowledge_distiller import KnowledgeDistillation
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt


DEVICE = 'cuda'
TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 200
NUM_WORKER = 8
EPOCHS = 50

EPSILON = np.append(np.arange(0.01, 1 / 60000, -0.001), 1 / 60000)
SPARSITY = np.arange(0, 1.01, 0.1)

ds_test_benign = PoisonedMNIST('.data', x_pixel_backdoor, epsilon=0, target=1, train=False, transform=transforms.Compose([transforms.ToTensor()]))
dl_test_benign = DataLoader(ds_test_benign, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)

ds_test_poisoned = PoisonedMNIST('.data', x_pixel_backdoor, epsilon=1, only_pd=True, target=1, train=False, transform=transforms.Compose([transforms.ToTensor()]))
dl_test_poisoned = DataLoader(ds_test_poisoned, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)

def test(model, dataset):
    correct = 0
    for i, (data, labels) in enumerate(dataset):
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        y_hat = model(data).max(1)[1]
        match = (y_hat == labels)
        correct += len(match.nonzero())
    return correct

def show_acc(result):
    fig = plt.figure(figsize=(8,8))
    plt.plot(EPSILON, [x[0] for x in result], "g*-", label="Benign")
    plt.plot(EPSILON, [x[1] for x in result], "ro-", label="Backdoored")
    # plt.yticks(np.arange(0, 1.1, step=0.01))
    plt.title("Benign vs. Backdoored")
    plt.xlabel("Percent of added poisoned data")
    plt.ylabel("Accuracy")
    plt.legend()
    return fig

def show_distill_acc(result):
    fig = plt.figure(figsize=(8,8))
    plt.plot(EPSILON, [x[0] for x in result], "r*--", label="Benign")
    plt.plot(EPSILON, [x[1] for x in result], "m*-", label="Teacher Backdoored")
    plt.plot(EPSILON, [x[2] for x in result], "yo--", label="Dis on Clean Benign")
    plt.plot(EPSILON, [x[3] for x in result], "go-", label="Dis on Clean Backdoored")
    plt.plot(EPSILON, [x[4] for x in result], "c^--", label="Dis on Poisoned Benign")
    plt.plot(EPSILON, [x[5] for x in result], "b^-", label="Dis on Poisoned Backdoored")
    # plt.yticks(np.arange(0, 1.1, step=0.01))
    plt.title("Distillation vs. epsilon")
    plt.xlabel("Percent of added poisoned data")
    plt.ylabel("Accuracy")
    plt.legend()
    return fig

def show_prun_acc(result):
    fig = plt.figure(figsize=(8,8))
    plt.plot(SPARSITY, [x[0] for x in result], "g*-", label="Benign")
    plt.plot(SPARSITY, [x[1] for x in result], "ro-", label="Backdoored")
    # plt.yticks(np.arange(0, 1.1, step=0.01))
    plt.title("Benign vs. Backdoored Under pruning")
    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")
    plt.legend()
    return fig

def epsilon_and_test():
    result = []
    for eps in EPSILON:
        model_path = os.path.join('.models', f'badnets_mnist_{(eps * 1000):.2f}.pth')
        model = MNISTNetwork().to(DEVICE)
        if False and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location = DEVICE))
        else:
            ds_train = PoisonedMNIST('.data', x_pixel_backdoor, epsilon=float(0.002), target=1, train=True, transform=transforms.Compose([transforms.ToTensor()]))
            dl_train = DataLoader(ds_train, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)
            t = Trainer(model, dl_train, DEVICE)
            t.train()
            # torch.save(model.state_dict(), model_path)
        
        model.eval()

        c1 = test(model, dl_test_benign)
        c2 = test(model, dl_test_poisoned)
        print(c1 / len(ds_test_benign), c2 / len(ds_test_poisoned))
        result.append((c1 / len(ds_test_benign), c2 / len(ds_test_poisoned)))
    return result

def distill_and_test(writer: SummaryWriter = None):
    def closure(teacher, student, ds):
        dis = KnowledgeDistillation(teacher, student, ds, DEVICE, temperature=100.)
        student = dis.distill()
        student.eval()
        clean = test(student, dl_test_benign)
        pd = test(student, dl_test_poisoned)
        return clean, pd

    result = []
    for step, eps in enumerate(EPSILON):
        model_path = os.path.join('.models', f'badnets_mnist_{(eps * 1000):.2f}.pth')
        teacher = MNISTNetwork().to(DEVICE)
        ds_train_pd = PoisonedMNIST('.data', x_pixel_backdoor, epsilon=float(eps), target=1, train=True, transform=transforms.Compose([transforms.ToTensor()]))
        dl_train_pd = DataLoader(ds_train_pd, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)
        ds_train_oc = PoisonedMNIST('.data', x_pixel_backdoor, epsilon=float(0), target=1, train=True, transform=transforms.Compose([transforms.ToTensor()]))
        dl_train_oc = DataLoader(ds_train_oc, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)
        if True and os.path.exists(model_path):
            teacher.load_state_dict(torch.load(model_path, map_location = DEVICE))
        else:
            t = Trainer(teacher, dl_train_pd, DEVICE)
            t.train()
            # torch.save(model.state_dict(), model_path)
        
        teacher.eval()

        student_oc = MNISTNetwork().to(DEVICE)
        student_pd = MNISTNetwork().to(DEVICE)
        
        pd1, pd2 = closure(teacher, student_pd, dl_train_pd)
        oc1, oc2 = closure(teacher, student_oc, dl_train_oc)

        c1 = test(teacher, dl_test_benign)
        c2 = test(teacher, dl_test_poisoned)
        print(f"{eps}: teacher: {c1 / len(ds_test_benign)}, {c2 / len(ds_test_poisoned)}")
        print(f"{eps}: student_pd: {pd1 / len(ds_test_benign)}, {pd2 / len(ds_test_poisoned)}")
        print(f"{eps}: student_oc: {oc1 / len(ds_test_benign)}, {oc2 / len(ds_test_poisoned)}")

        if writer is not None:
            r = {
                "teacher_benign": c1/len(ds_test_benign),
                "teacher_poisoned": c2/len(ds_test_poisoned),
                "student_oc_benign": oc1/len(ds_test_benign),
                "student_oc_poisoned": oc2/len(ds_test_poisoned),
                "student_pd_benign": pd1/len(ds_test_benign),
                "student_pd_poisoned": pd2/len(ds_test_poisoned),
            }
            writer.add_scalars('distillation', r, step)
        result.append((c1 / len(ds_test_benign), c2 / len(ds_test_poisoned), oc1 / len(ds_test_benign), oc2 / len(ds_test_poisoned), pd1 / len(ds_test_benign), pd2 / len(ds_test_poisoned)))
    return result



def prune_and_test(model_path):
    result = []
    for sp in SPARSITY:
        model = MNISTNetwork().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        pruner = MagnitudePruner(model, fine_tuning=False, sparsity=sp)
        model, mask = pruner.prune()
        model.eval()
        c1 = test(model, dl_test_benign)
        c2 = test(model, dl_test_poisoned)
        print(c1 / len(ds_test_benign), c2 / len(ds_test_poisoned))
        result.append((c1 / len(ds_test_benign), c2 / len(ds_test_poisoned)))

    return result

def prune_with_fine_tuning_on_posioned_data(model_path):
    result = []
    ds_train = PoisonedMNIST('.data', x_pixel_backdoor, epsilon=float(0.002), target=1, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    dl_train = DataLoader(ds_train, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)
    trainer = {
        "device": DEVICE,
        "batch_size": 500,
        "epochs": 5,
        "criterion": torch.nn.CrossEntropyLoss(),
        "optimizer": lambda m : torch.optim.SGD(m, lr=0.01, momentum=0.9),
        "dataset": dl_train
    }

    for sp in SPARSITY:
        model = MNISTNetwork().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        pruner = MagnitudePruner(model, fine_tuning=True, trainer=trainer, pruning_steps=50, sparsity=sp, log_steps=True)
        model, mask = pruner.prune()
        model.eval()
        c1 = test(model, dl_test_benign)
        c2 = test(model, dl_test_poisoned)
        print(c1 / len(ds_test_benign), c2 / len(ds_test_poisoned))
        result.append((c1 / len(ds_test_benign), c2 / len(ds_test_poisoned)))

    return result

def prune_with_fine_tuning_on_clean_data(model_path):
    result = []
    ds_train = PoisonedMNIST('.data', x_pixel_backdoor, epsilon=0, target=1, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    dl_train = DataLoader(ds_train, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)
    trainer = {
        "device": DEVICE,
        "batch_size": 500,
        "epochs": 5,
        "criterion": torch.nn.CrossEntropyLoss(),
        "optimizer": lambda m : torch.optim.SGD(m, lr=0.01, momentum=0.9),
        "dataset": dl_train
    }

    for sp in SPARSITY:
        model = MNISTNetwork().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        pruner = MagnitudePruner(model, fine_tuning=True, trainer=trainer, pruning_steps=50, sparsity=sp, log_steps=True)
        model, mask = pruner.prune()
        model.eval()
        c1 = test(model, dl_test_benign)
        c2 = test(model, dl_test_poisoned)
        print(c1 / len(ds_test_benign), c2 / len(ds_test_poisoned))
        result.append((c1 / len(ds_test_benign), c2 / len(ds_test_poisoned)))

    return result

writer = SummaryWriter(f"runs/badnets_test_{datetime.now().strftime('%Y_%m_%d_%H_%M')}")
# result = epsilon_and_test()
# writer.add_figure('bvb', show_acc(result), global_step=0)
# result = prune_with_fine_tuning_on_posioned_data(os.path.join('.models', 'badnets_mnist_2.00.pth'))
# writer.add_figure('bvb_up_ft_pd', show_prun_acc(result), global_step=0)
# result = prune_with_fine_tuning_on_clean_data(os.path.join('.models', 'badnets_mnist_2.00.pth'))
# writer.add_figure('bvb_up_ft_oc', show_prun_acc(result), global_step=0)
result = distill_and_test(writer)
show_distill_acc(result)
writer.close()
