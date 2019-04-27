#!/bin/bash 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Downloading and unpacking SVHN"
IMGDIR="$DIR/../images/svhn"
LABELDIR="$DIR/../labels/svhn"
mkdir -p $IMGDIR
mkdir -p $LABELDIR
python3 $DIR/unpack_svhn.py $IMGDIR $LABELDIR
