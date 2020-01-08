for eps in 0.0005 0.00002
do
    python badnets_cifar10_resnet.py --gpu=cuda:3 --eps=$eps
done
