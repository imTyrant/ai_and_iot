#!/bin/bash

gpu=cuda:2
declare -a EPS=(0.002 0.001 0.0005 0.00002)

# check target dir exist
if [ ! -d ".models/cifar10_resnet" ]; then
    mkdir .models/cifar10_resnet
fi

# firstly we train models
for eps in ${EPS[@]}; do
    python badnets_cifar10_resnet.py --nobd=0 --gpu=$gpu --mode=tp --eps=$eps
done

# then we test weight pruning
for eps in ${EPS[@]}; do
    for sp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95; do
        python badnets_cifar10_resnet.py --nobd=0 --gpu=$gpu --mode=prune --eps=$eps --sp=$sp
    done
done

# distillation
for eps in ${EPS[@]}; do
    python badnets_cifar10_resnet.py --mode=distill --gpu=$gpu --eps=$eps --nobd=0
done
