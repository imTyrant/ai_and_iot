#!/bin/bash

# git submodule init && git submodule update

# if [ ! -d .models ]; then
#     mkdir .models
# fi

# if [ ! -d .data ]; then
#     mkdir .data
# fi

# download data
cd .data
while [ true ]; do
    echo "------------------------"
    echo "Select Datset..."
    echo "1: MNIST"
    echo "2: CIFAR10"
    echo "3: SVHN"
    echo "4: VGG-Face"
    echo "A: All"
    echo "Q: Do nothing"

    read -p "Selecting: " ops
    case $ops in
        1)
            echo "Downloading MNIST.."
        ;;
        2)
            echo "Downloading CIFAR10.."
        ;;
        3)
            echo "Downloading SVHN.."
        ;;
        4)
            echo "Downloading VGG-Face.."
            if [ ! -d vgg_face ]; then
                mkdir vgg_face
            fi
            cd vgg_face
            wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz
            cd ..
        ;;
        a|A)
            echo "Downloading all of datasets"
        ;;
        q | Q)
            echo "Do nothing"
            break
        ;;
        *)
            echo "Invalid input"
        ;;
    esac
done
cd ..