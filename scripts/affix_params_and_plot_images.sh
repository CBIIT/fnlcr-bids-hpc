#!/bin/bash

# Run like /data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc/scripts/affix_params_and_plot_images.sh "/home/weismanal/notebook/2019-04-26/test_jurgens_script_using_candle/last-exp"

# Input is the experiment directory, e.g., "/home/weismanal/notebook/2019-04-26/test_jurgens_script_using_candle/last-exp" or "/gpfs/gsfs10/users/weismanal/notebook/2019-04-26/test_jurgens_script_using_candle/experiments/X007"
expdir=$1

# Constant; we're assuming all images are .png files
suffix=".png"

# Create a single directory that will contain descriptively named links to all the images
linksdir=${expdir}/links
mkdir $linksdir

# For each hyperparameter set...
for dir in ${expdir}/run/*; do

    # Get the hyperparameters used from the model.log file
    tmp=$(awk 'BEGIN{doprint=0} {if(doprint==1 && $0~/^$/){doprint=0;print ""}; if(doprint)printf("__%s-%s",$1,$2); if($0~/^PARAMS:$/)doprint=1}' $dir/model.log)
    hpstr=${tmp:2:${#tmp}} #pretty=$(echo $hpstr | awk '{gsub("__"," "); gsub("-","="); print}')

    # Determine the base name of the image    
    bn=$(basename $dir/*${suffix} "${suffix}")

    # Make a descriptively named link in the linksdir folder to the image
    ln -s ${dir}/${bn}${suffix} ${linksdir}/${bn}--${hpstr}${suffix}

done

# Use the make_image_gallery.sh script to downsample the images and create a gallery of them
/data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc/scripts/make_image_gallery.sh $linksdir "${suffix}" 480 0 "lanczos" 3 "${expdir}"