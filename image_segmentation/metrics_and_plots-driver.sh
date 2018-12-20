#!/bin/bash

# Run like:
#
#   ./metrics_and_plots-driver.sh &> out_and_err.txt
#

# Parameters
produce_metrics=1
produce_plots=1
framerate=2 # frames per second
movie_dir=/Users/weismanal/notebook/2018-12-12/movies
#models="01-roi1_only_uncombined_unet 03-roi1+roi2_uncombined_unet"
models=$(find . -type d -iname "0*" | awk -v FS="./" '{print $2}' | sort)

# Run the main python script for each model
for model in $models; do
    python produce_metrics_and_plots.py $model $produce_metrics $produce_plots
done

# If we've created images, make movies of them
if [ $produce_plots -eq 1 ]; then

    for model in $models; do
	
	# For every directory of images that we can find...
	for images_dir in $(find $model -type d -iname "movie-*" | grep -v "backup\|archive" | sort); do
	    pushd $images_dir > /dev/null # enter the directory
	    bn=$(echo $images_dir | awk '{gsub("/movie",""); print}')
	    fn=${movie_dir}/${bn}.mp4 # set the movie filename
	    ffmpeg -r $framerate -pattern_type glob -i "*.png" -c:v libx264 -crf 23 -profile:v baseline -level 3.0 -pix_fmt yuv420p -c:a aac -ac 2 -b:a 128k -movflags faststart $fn # create a movie that should work in all browsers
	    popd > /dev/null # go back to the top directory
	done

    done
fi
