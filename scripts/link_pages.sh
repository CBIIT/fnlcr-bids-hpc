#!/bin/bash

# Constant
root="$HOME/checkouts/fnlcr-bids-hpc"

# Function to create paddings based on depth of current directory
function pad_str() {
    str0=$1
    ntimes=$2
    itime=0
    str=""
    while [ $itime -lt "$ntimes" ]; do
        str="${str}${str0}"
        itime=$((itime+1))
    done
    echo "$str"
}

# Function to print a single table of contents
function get_curr_toc() {
    dirlist=$1
    nroot=$2
    id1=$3
    basepath=$4
    id2=0
    echo "## Site Directory"
    for dir in $dirlist; do
        readme="$dir/README.md"
        dirname=$(grep "^#\ " "$readme" | head -n 1 | awk -v FS="# " '{print $2}')
        ntimes=$(echo "$dir" | awk '{print(gsub("/","/"))}')
        ntimes=$((ntimes-nroot))
        if [ "$id1" == "$id2" ]; then
            str="**${dirname}**"
        else
            link=https://cbiit.github.io$(echo "$dir" | awk -v FS="$basepath" '{print $2}')
            str="[$dirname]($link)"
        fi
        echo "$(pad_str "  " $ntimes)* $str"
        id2=$((id2+1))
    done
    echo -e "\n\n---\n"
}

# Variables
nroot=$(echo "$root" | awk '{print(gsub("/","/"))}')
dirlist=$(find "$root" -type d ! -regex ".*\.git/*.*" ! -regex ".*__pycache__.*" | sort)
basepath=$(dirname "$root")

# Create README.md files for each directory that doesn't already contain one
for dir in $dirlist; do
    single_dir=$(basename "$dir")
    readme="$dir/README.md"
    if [ ! -f "$readme" ]; then
        echo "# $single_dir" > "$readme"
        echo "README created in $dir"
    fi
done

# For each directory...
id1=0
for dir0 in $dirlist; do

    # Determine the corresponding README file
    readme0="$dir0/README.md"

    # Rename the current README file
    cp "$readme0" "$dir0/README-tmp.md"

    # Output the current TOC into a new README file
    get_curr_toc "$dirlist" "$nroot" "$id1" "$basepath" > tmp.txt

    # Append the original contents of the README file to the new README file
    awk -v doprint=0 '{if(!doprint){if($0~"^# ")doprint=1}; if(doprint)print}' "$dir0/README-tmp.md" >> tmp.txt

    # Remove the old README
    mv -f tmp.txt "$readme0"
    rm -f "$dir0/README-tmp.md"

    # Get the ID of the next directory
    id1=$((id1+1))

done
