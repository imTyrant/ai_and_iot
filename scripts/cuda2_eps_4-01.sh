for eps in 0.002 0.001
do
    python badnet_cifar10_resnet.py --gpu=cuda:2 --eps=$eps
done
