#!/bin/bash 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#echo "Downloading and unpacking MNIST"
IMGDIR="$DIR/../images/mnist"
LABELDIR="$DIR/../labels/mnist"
mkdir -p $IMGDIR
mkdir -p $LABELDIR
python3 $DIR/unpack_mnist.py $IMGDIR $LABELDIR
