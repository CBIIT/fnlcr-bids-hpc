#!/bin/bash

function make_inference_params_file() {
    # From the last good training parameters file and set of weights, create the corresponding inference parameters file
    weightsfile=$1
    paramsfile=$2
    inf_images=$3
    awk -v doprint=1 '{if($0~/^verbose=/)doprint=0; if(doprint)print}' $paramsfile | grep -v "^images *=\|^labels *=\|^initialize *=\|^predict *=\|^epochs *=\|^encoder *=\|^batch_size *=\|^obj_return *="
    echo -e "images = '$inf_images'\ninitialize = '$weightsfile'\npredict = True"
}



FNLCR="/home/weismanal/checkouts/fnlcr-bids-hpc"
cd params_files
ijob=0
for file in *; do
    ijob=$[ijob+1]
    ijob2=$(printf '%03i' $ijob)
    jobdir="job_${ijob2}"
    mkdir ../inference_jobs/$jobdir
    pushd ../inference_jobs/$jobdir
    mv ../../params_files/$file .
    cp $FNLCR/candle/run_without_candle-template.sh run_without_candle.sh
    $file | grep resnet && model="resnet" || model="unet"
    awk -v jobname=$jobdir -v paramsfile=$file -v model="${model}.py" '{gsub("12:00:00","00:45:00"); gsub("hpset_23",jobname); gsub("/home/weismanal/notebook/2019-01-28/jobs/not_candle/single_param_set.txt",paramsfile); gsub("unet.py",model); print}' $FNLCR/candle/run_without_candle-template.sh > run_without_candle.sh
    popd
done