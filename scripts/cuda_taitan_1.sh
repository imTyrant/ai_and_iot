gpu=cuda:1
for eps in 0.002 0.001 0.0005 0.00002
do
    for sp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
    do 
        python badnets_cifar10_resnet.py --mode=prune --gpu=$gpu --sp=$sp --eps=$eps --nobd=0
    done
done
