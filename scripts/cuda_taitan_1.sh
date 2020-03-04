gpu=cuda:1

function valid () {
    if [ $? -ne 0 ]; then
        exit
    fi
}

for eps in 0.002 0.001 0.0005 0.00002; do
    python badnets_svhn.py --gpu=$gpu --nobd=0 --mode=tp --eps=$eps
    valid
    for sp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95; do
        python badnets_svhn.py --gpu=$gpu --nobd=0 --mode=prune --eps=$eps --sp=$sp
        valid
    done
done

python badnets_cifar10_resnet.py --gpu=$gpu --nobd=0 --mode=tp --eps=0.01
valid

for sp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95; do
    python badnets_cifar10_resnet.py --gpu=$gpu --nobd=0 --mode=prune --eps=0.01 --sp=$sp
    valid
done
