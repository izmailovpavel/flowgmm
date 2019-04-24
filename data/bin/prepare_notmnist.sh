#!/bin/bash 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Downloading and unpacking NotMNIST"
IMGDIR="$DIR/../images/notmnist"
mkdir -p $IMGDIR
curl -o $IMGDIR/notMNIST_small.tar.gz http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
gunzip $IMGDIR/notMNIST_small.tar.gz
tar -xf $IMGDIR/notMNIST_small.tar -C $IMGDIR
