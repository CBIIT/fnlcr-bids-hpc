#!/bin/bash

function make_inference_params_file() {
    # From the last good training parameters file and set of weights, create the corresponding inference parameters file
    weightsfile=$1
    paramsfile=$2
    inf_images=$3
    awk -v doprint=1 '{if($0~/^verbose=/)doprint=0; if(doprint)print}' $paramsfile | grep -v "^images *=\|^labels *=\|^initialize *=\|^predict *=\|^epochs *=\|^encoder *=\|^batch_size *=\|^obj_return *="
    echo -e "images = '$inf_images'\ninitialize = '$weightsfile'\npredict = True"
}



IMAGE_SEG=/home/weismanal/checkouts/fnlcr-bids-hpc/image_segmentation
#IMAGE_SEG=/Users/weismanal/checkouts/fnlcr-bids-hpc/image_segmentation
module load python/3.6 FFmpeg
# ln -s ../inference/inference_images/roi3_prepared-z_first.npy roi3_input_img.npy
# ln -s ~/links/1-pre-processing/roi3/1-not_padded/roi3_masks_original.npy known_masks_roi3.npy
# for file in ../inference/inference_jobs/*; do
#     jobnum=$(basename $file | awk -v FS="_" '{print $2}')
#     hpset=$(basename $(grep DEFAULT_PARAMS_FILE ../inference/inference_jobs/job_${jobnum}/*.sh | awk -v FS="DEFAULT_PARAMS_FILE=" '{print $2}') | awk -v FS="_" '{print $2}')
#     dir=$(basename $file/*.txt | rev | awk -v FS="txt." '{print $2}' | awk -v FS="_" '{print $1}' | rev)
#     roi=$(basename $file/*.txt | rev | awk -v FS="txt." '{print $2}' | awk -v FS="_" '{print $2}' | rev)
#     model_dir="hpset_${hpset}"
#     mkdir -p $model_dir
#     pushd $model_dir
#     ln -s $(ls ../$file/*.npy) "inferred_masks-${roi}-${dir}_first.npy"
#     popd
# done

#imodel=0; for model in $(find . -mindepth 1 -maxdepth 1 -type d | sort | awk -v FS="./" '{print $2}'); do imodel=$[imodel+1]; imodel2=$(printf '%02i' $imodel); mv $model "$imodel2-$model"; done

#find . -mindepth 1 -maxdepth 1 -type d | sort | awk -v FS="./" -v ORS=" " '{print $2}'

#models=$(find . -mindepth 1 -maxdepth 1 -type d | sort | awk -v FS="./" -v ORS=" " '{print $2}')
#models=andrew
models=$(find . -mindepth 1 -maxdepth 1 -type d | sort | awk -v FS="./" -v ORS=" " '{print $2}')
#models="02-hpset_11"
roi_numbers=[3]
ninferences=3
calculate_metrics=1
create_plots=1
#movie_directory=/Users/weismanal/biowulf-data/notebook/2019-02-11/postprocessing/movies
movie_directory=/home/weismanal/notebook/2019-02-11/postprocessing/movies
nframes=40
framerate=2
#$IMAGE_SEG/metrics_and_plots-driver.sh <MODELS> <ROI-NUMBERS> <NINFERENCES> <CALCULATE-METRICS> <CREATE-PLOTS> <MOVIE-DIRECTORY> <NFRAMES> <FRAMERATE>
$IMAGE_SEG/metrics_and_plots-driver.sh "$models" $roi_numbers $ninferences $calculate_metrics $create_plots $movie_directory $nframes $framerate
