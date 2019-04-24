#!/bin/bash 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Downloading and unpacking NotMNIST"
IMGDIR="$DIR/../images/notmnist"
mkdir -p $IMGDIR
curl -o $IMGDIR/notMNIST_small.tar.gz http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
gunzip $IMGDIR/notMNIST_small.tar.gz
tar -xf $IMGDIR/notMNIST_small.tar -C $IMGDIR
for labeldir in $IMGDIR/notMNIST_small/*
do
    i=0
    for img in $labeldir/*
    do
        ((i+=1))
        mv $img ${labeldir}/${i}.png
    done
done
#Pillow can not load these two files for some reason
rm $IMGDIR/notMNIST_small/A/1288.png
rm $IMGDIR/notMNIST_small/F/417.png
