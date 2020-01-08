for eps in 0.01 0.008 0.006 0.004 0.002 0.001 0.0005 0.00002
do
    python badnets_cifar10_resnet.py --gpu=cuda:0 --eps=$eps
done
