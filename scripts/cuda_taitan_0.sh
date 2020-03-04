gpu=cuda:0

function valid () {
    if [ $? -ne 0 ]; then
        exit
    fi
}

for eps in 0.01 0.008 0.006 0.004; do
    python badnets_svhn.py --gpu=$gpu --nobd=0 --mode=tp --eps=$eps
    valid
    for sp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95; do
        python badnets_svhn.py --gpu=$gpu --nobd=0 --mode=prune --eps=$eps --sp=$sp
        valid
    done
done

for sp in 0.1 0.2 0.3 0.4 0.5; do
    python badnets_cifar10_resnet.py --gpu=$gpu --nobd=0 --mode=prune --eps=0.01 --sp=$sp
    valid
done
