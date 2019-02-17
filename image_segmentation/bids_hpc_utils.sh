#!/bin/bash




#models=$(find . -mindepth 1 -maxdepth 1 -type d | sort | awk -v FS="./" -v ORS=" " '{print $2}')
models=$(find . -mindepth 1 -maxdepth 1 -type d -iregex "^\./1[2-7].*" | sort | awk -v FS="./" -v ORS=" " '{print $2}')
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
