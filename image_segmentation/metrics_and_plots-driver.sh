#!/bin/bash

# Define a function to set default values to the parameters if they're blank, ""
function process_arg(input,default) {
    if [ "a${input}" != "a" ]; then
	echo $input
    else
	echo $default
    fi
}

# Process the parameters to this script
models=process_arg($1,"my_model")
roi_nums=process_arg($2,"[1,2,3]")
ninf=process_arg($3,3)
calculate_metrics=process_arg($4,0)
create_plots=process_arg($5,0)
movie_dir=process_arg($6,"movies")
nframes=process_arg($7,40)
framerate=process_arg($8,2)

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# Run the main python script for each model
for model in $models; do
    python ${SCRIPT_DIR}/metrics_and_plots.py $model $roi_nums $ninf $calculate_metrics $create_plots $nframes
done

# If we've created images, make movies of them
if [ $create_plots -eq 1 ]; then

    # Create the movie directory if it's not already present
    if [ ! -d $movie_dir ]; then
	mkdir $movie_dir
    fi

    # For each model...
    for model in $models; do
	
	# For every directory of images that we can find...
	for images_dir in $(find $model -type d -iname "movie-*" | grep -v "backup\|archive" | sort); do

	    # Enter the current directory of images
	    pushd $images_dir > /dev/null

	    # Set the movie filename
	    fn=${movie_dir}/$(echo $images_dir | awk '{gsub("/movie",""); print}').mp4

	    # Create a movie that should work in all browsers
	    ffmpeg -r $framerate -pattern_type glob -i "*.png" -c:v libx264 -crf 23 -profile:v baseline -level 3.0 -pix_fmt yuv420p -c:a aac -ac 2 -b:a 128k -movflags faststart $fn

	    # Enter the working directory again
	    popd > /dev/null
	    
	done

    done
fi
