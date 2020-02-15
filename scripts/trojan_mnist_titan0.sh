#!/bin/bash

gpu=cuda:0

declare -a EPS=(1.0 0.01 0.009 0.008 0.007 0.006 0.005 0.004 0.003 0.002 0.001 0.00002)

for eps in ${EPS[@]}; do
    python trojan_mnist.py --mode=tp --eps=$eps --gpu=$gpu
done