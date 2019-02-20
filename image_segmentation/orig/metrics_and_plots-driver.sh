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

		#pwd

	    # Create a movie that should work in all browsers
	    ffmpeg -r $framerate -pattern_type glob -i "*.png" -c:v libx264 -crf 23 -profile:v baseline -level 3.0 -pix_fmt yuv420p -c:a aac -ac 2 -b:a 128k -movflags faststart $fn

	    # Enter the working directory again
	    popd > /dev/null
	    
	done

    done
fi