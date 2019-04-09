#!/bin/bash

root="$HOME/checkouts/fnlcr-bids-hpc"

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


id1=0
for dir0 in $dirlist; do
    readme0="$dir0/README.md"
    echo $readme0

    id2=0
    for dir in $dirlist; do
        readme="$dir/README.md"
        dirname=$(grep "^#\ " "$readme" | head -n 1 | awk -v FS="# " '{print $2}')
        ntimes=$(echo "$dir" | awk '{print(gsub("/","/"))}')
        ntimes=$((ntimes-nroot))
        if [ $id1 -eq $id2 ]; then
            str="**${dirname}**"
        else
            link=https://cbiit.github.io$(echo "$dir" | awk -v FS="$basepath" '{print $2}')
            str="[$dirname]($link)"
        fi
        echo "$(pad_str "  " $ntimes)* $str"
        id2=$((id2+1))
    done

    id1=$((id1+1))
done






# * **Home**
#   * [Image segmentation](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation)
#   * [Documentation](https://cbiit.github.io/fnlcr-bids-hpc/documentation)
#     * [How to create a central git repository](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_create_a_central_git_repo)



# * [Home](https://cbiit.github.io/fnlcr-bids-hpc)
#   * [Image segmentation](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation)
#   * **Documentation**
#     * [How to create a central git repository](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_create_a_central_git_repo)