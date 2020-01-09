#!/bin/bash
# require gpu
gpu=$1
if [ $1 ]; then 
    echo "Name $1"
else
    echo "You should name a gpu like cuda:0"
    exit 1
fi

# make model dir

if [ ! -d ".models/$gpu" ]; then
    mkdir .models/$gpu
fi

# activate conda env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pt1.3.0

# do works
for eps in 0.01 0.008 0.006 0.004 0.002 0.001 0.0005 0.00002
do
    python badnets_cifar10_resnet.py --gpu=$gpu --eps=$eps --nobd=0 --mp=$gpu
done

# clean up
conda deactivate

