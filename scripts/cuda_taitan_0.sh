gpu=cuda:0
for eps in 0.01 0.008 0.006 0.004
do
    for sp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
    do 
        python badnets_cifar10_resnet.py --mode=prune --gpu=$gpu --sp=$sp --eps=$eps --nobd=0
    done
done
