for eps in 0.01 0.008 0.006 0.004
do
    python badnet_cifar10_resnet.py --gpu=cuda:0 --eps=$eps
done
