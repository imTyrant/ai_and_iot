gpu=cuda:1
for eps in 0.002 0.001 0.0005 0.00002
do
    python badnets_cifar10_resnet.py --mode=distill --gpu=$gpu --eps=$eps --nobd=0
done
