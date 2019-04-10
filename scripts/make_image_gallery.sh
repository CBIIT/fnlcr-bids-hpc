#!/bin/bash

# Call like:
#
#   make_image_gallery IMAGE-DIR FILETYPE WIDTH VERBOSE FILTERS NUM-COLUMNS OUTPUT-DIR
#
# e.g.,
#
#   make_image_gallery "/home/weismanal/notebook/2019-02-11/postprocessing/13-hpset_32/movie-roi3_view_x" ".png" 480 0 "point cubic lanczos mitchell" 3 "/home/weismanal/notebook/2019-04-09"
#

# Use ImageMajick's convert command to resize each image using a particular filter
function convert_with_filter() {
    # Define paramters
    dirname=$1
    bn=$2
    suffix=$3
    filter=$4
    size=$5
    verbose=$6

    # Run the ImageMajick function "convert"
    convert "${dirname}/${bn}${suffix}" -filter "$filter" -define filter:verbose="$verbose" -resize "${size}" "${dirname}/${bn}-${filter}${suffix}"
}

function write_webpage() {
    # Define parameters
    filter=$1
    images_dir=$2
    ncol=$3
    dirname=$4
    suffix=$5
    size=$6
    verbose=$7
    output_dir=$8

    # HTML header
    echo "<html>"
    echo "  <head>"
    echo "    <title>${filter}</title>"
    echo "  </head>"
    echo "  <body>"
    echo ""

    # Output current settings
    echo "<b>Settings:</b>"
    echo "<ul>"
    echo "<li>filter: $filter</li>"
    echo "<li>dirname: $dirname</li>"
    echo "<li>suffix: $suffix</li>"
    echo "<li>size: $size</li>"
    echo "<li>verbose: $verbose</li>"
    echo "<li>ncol: $ncol</li>"
    echo "<li>output_dir: $output_dir</li>"
    echo "</ul>"

    # Write the table of images
    echo "<table border=1>"
    icol=1
    echo "  <tr>"
    iimg=1
    nimg=$(find "$images_dir" -type f | wc -l)
    for image in "${images_dir}"/*; do
        image_name=$(basename "$image")
        echo -e "    <td align=\"center\">${image_name}\n<img src=\"images-${filter}/${image_name}\"></td>"
        if [ "$iimg" -lt "$nimg" ]; then
            if [ "$icol" -lt "$ncol" ]; then
                icol=$((icol+1))
            else
                echo "  </tr>"
                icol=1
                echo "  <tr>"
            fi
            iimg=$((iimg+1))
        else
            echo "  </tr>"
        fi
    done
    echo "</table>"

    # HTML footer
    echo ""
    echo "  </body>"
    echo "</html>"
}

# Inputs
dirname=$1 #"/home/weismanal/notebook/2019-02-11/postprocessing/13-hpset_32/movie-roi3_view_x" # directory containing images
suffix=$2 #".png" # image suffix
size=$3 #480 # e.g., 50% or 640 (can be percent or width in pixels)
verbose=$4 #0 # whether to display the ImageMajick processing details
filters=$5 #"point cubic lanczos mitchell" # filters to try
ncol=$6 #3 # number of columns (of width ~$size [if in pixels]) in the table of images
output_dir=$7 #"/home/weismanal/notebook/2019-04-09" # directory in which to output the resized images and the webpages

# Load the ImageMagick module
module load ImageMagick/7.0.8

# For each image in the image directory...
for file in "${dirname}/"*"${suffix}"; do

    # Output status
    echo "Processing ${file}..."

    # Get the "basename" of the image (i.e., without the path and without the suffix)
    bn=$(basename "$(basename "$file")" "$suffix") # e.g., bn="frame_0001"

    # For each filter type...
    for filter in $filters; do

        # Resize the image using the current filter type
        convert_with_filter "$dirname" "$bn" "$suffix" "$filter" "$size" "$verbose"

    done

done

# For each filter type...
for filter in $filters; do

    # Obtain the output image directory, create it, and move all the corresponding images to there
    images_dir="${output_dir}/images-$filter"
    mkdir "$images_dir"
    mv "${dirname}/"*"-${filter}${suffix}" "$images_dir"

    # Write the current webpage
    write_webpage "$filter" "$images_dir" "$ncol" "$dirname" "$suffix" "$size" "$verbose" "$output_dir" > "${output_dir}/gallery-${filter}.html"

done