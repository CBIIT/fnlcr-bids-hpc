#!/bin/bash

# Run this script like:
#
#   affix_params_and_plot_images.sh SUFFIX WIDTH NCOLUMNS [EXPDIR]
#
# Arguments:
#
#   (1) image suffix, e.g., ".png" or "umap_pbmck3.png"
#   (2) desired image width in pixels, e.g., 480
#   (3) number of columns in image gallery, e.g., 3
#   (4) OPTIONAL: experiment directory, e.g., /home/weismanal/notebook/2019-04-26/test_jurgens_script_using_candle/last-exp or /gpfs/gsfs10/users/weismanal/notebook/2019-04-26/test_jurgens_script_using_candle/experiments/X007 or last-exp
#
# Sample calls:
#
#   /data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc/scripts/affix_params_and_plot_images.sh ".png" 480 3
#   /data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc/scripts/affix_params_and_plot_images.sh ".png" 480 3 last-exp
#   /data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc/scripts/affix_params_and_plot_images.sh ".png" 480 3 /home/weismanal/notebook/2019-04-26/test_jurgens_script_using_candle/last-exp
#   /data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc/scripts/affix_params_and_plot_images.sh "umap_pbmck3.png" 480 3
#

# Process the script arguments
suffix=$1
width=$2
ncolumns=$3
expdir=$4

# Allow for argument to be relative path, full path, and current directory... point is the argument must contain the "run" directory from the CANDLE job
if [ -z "$expdir" ]; then
    expdir=$(pwd)
elif [ "a${expdir:0:1}" != "a/" ]; then
    expdir=$(pwd)/$expdir
fi

# Ensure the input argument contains the 'run' directory, i.e., is an experiment directory
if [ ! -d "$expdir/run" ]; then
    echo "Error: Input experiment directory ($expdir) must contain the 'run' directory"
    exit 1
fi

# Create a single directory that will contain descriptively named links to all the images
linksdir=${expdir}/links
mkdir "$linksdir"

# For each hyperparameter set...
for dir in ${expdir}/run/*; do

    # Get the hyperparameters used from the model.log file
    tmp=$(awk 'BEGIN{doprint=0} {if(doprint==1 && $0~/^$/){doprint=0;print ""}; if(doprint)printf("__%s-%s",$1,$2); if($0~/^PARAMS:$/)doprint=1}' "$dir/model.log")
    hpstr=${tmp:2:${#tmp}} #pretty=$(echo $hpstr | awk '{gsub("__"," "); gsub("-","="); print}')

    # For each image in the folder describing the current HP set...
    for image in $dir/*${suffix}; do

        # Allow for full filenames, e.g., "umap_pbmck3.png", to be set as the suffix; obtain the actual filetype
        suffix2=".$(echo "$suffix" | awk -v FS="." '{print $2}')"

        # Determine the base name of the image    
        bn=$(basename "$image" "$suffix2")

        # Make a descriptively named link in the linksdir folder to the image
        ln -s "${dir}/${bn}${suffix2}" "${linksdir}/${hpstr}--${bn}${suffix2}"

    done

done

# Use the make_image_gallery.sh script to downsample the images and create a gallery of them
/data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc/scripts/make_image_gallery.sh "$linksdir" "$suffix2" "$width" 0 "lanczos" "$ncolumns" "$expdir"