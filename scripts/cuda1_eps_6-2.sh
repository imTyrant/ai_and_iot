for eps in 0.006 0.004
do
    python badnet_cifar10_resnet.py --gpu=cuda:1 --eps=$eps
done
