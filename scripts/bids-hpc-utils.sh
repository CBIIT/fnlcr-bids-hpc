#!/bin/bash

inf_images="andrew.npy"

IFS=$'\n'
for line in $(awk '{print $1, $2}' rundirs.txt | awk -v FS=";" '{print $1}'); do
    hpset=$(echo $line | awk '{print $1}')
    dir=$(echo $line | awk '{print $2}')
    echo "**** $dir ****"
    nmatches=$(find $dir -type f -iname "*.h5" | grep -v archive | wc -l) #| grep "hpset_$hpset"
    if [ $nmatches -gt 1 ]; then
        weightsfile=$(find $dir -type f -iname "*.h5" | grep -v archive | grep "hpset_$hpset")
    else
        weightsfile=$(find $dir -type f -iname "*.h5" | grep -v archive)
    fi
    outputdir=$(dirname $weightsfile)
    paramsfile=$(ls $outputdir/*.txt | grep -v "result.txt\|tmp.txt")
    awk -v doprint=1 '{if($0~/^verbose=/)doprint=0; if(doprint)print}' $paramsfile | grep -v "^images *=\|^labels *=\|^initialize *=\|^predict *=\|^epochs *=\|^encoder *=\|^batch_size *=" > tmp.txt
    echo -e "images = '$inf_images'\ninitialize = '$weightsfile'\npredict = True" >> tmp.txt
    less tmp.txt
done