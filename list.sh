#!/bin/bash
rootdir=/home/tor/rsc/ml-foundation
dlist[0]=book/ebook
dlist[1]=course/ecourse
dlist[2]=talk/etalk
dlist[3]=method/emethod

for dir in "${dlist[@]}"
do
    echo '>>> listing: '$dir
    ls -1 -R $rootdir/$dir > $rootdir/$dir/LIST.txt
done

