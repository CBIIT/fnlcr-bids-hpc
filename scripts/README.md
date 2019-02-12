# Installation

Clone the current repository and add the scripts directory to your path, e.g.,

```bash
export PYTHONPATH=$PYTHONPATH:/home/weismanal/checkouts/fnlcr-bids-hpc/scripts
. /home/weismanal/checkouts/fnlcr-bids-hpc/scripts/bids_hpc_utils.sh
```

# Pre-inference workflow

**Prerequisites:** It is assumed you have completed training all the models and, e.g., recorded the results in an Excel spreadsheet.

## (0) Setup

In some working directory, create the necessary directories:

```bash
mkdir inference_images params_files inference_jobs
```

## (1) Preprocess the inference images

E.g., in Python:

```python
# Load pertinent modules
import numpy as np
import bids_hpc_utils as bhu

# Set parameters
images_npy_file = '/home/weismanal/links/1-pre-processing/roi3/1-not_padded/roi3_images_original.npy'
#images_npy_file = '/home/weismanal/links/1-pre-processing/field_of_view/field_of_view.npy'
idtype = 2
n_layers_max = 6

# Load images
images = np.load(images_npy_file)

# Automatically scale the images
images = bhu.normalize_images(images,idtype)

# Make the other two dimensions of the images accessible
x_first,y_first,z_first = bhu.transpose_stack(images)

# Pad the each of 2D planes of the images
x_first = bhu.pad_images(x_first,n_layers_max)
y_first = bhu.pad_images(y_first,n_layers_max)
z_first = bhu.pad_images(z_first,n_layers_max)

# Write these other "views" of the test data to disk
np.save('inference_images/roi3_prepared-x_first.npy',x_first)
np.save('inference_images/roi3_prepared-y_first.npy',y_first)
np.save('inference_images/roi3_prepared-z_first.npy',z_first)
```

This code prepares raw images (in .npy format) for inference by scaling them automatically (both the U-Net and ResNet Python scripts expect images in the uint16 format; this is set via `idtype=2`), splitting them into all three dimensions, and padding each of the resulting three stacks of images (in order for the images to be run through a U-Net of some number of layers; this is set via, e.g., `n_layers_max=6`).

## (2) Create the inference parameters files

Copy the ID and output directory columns of the Excel spreadsheet to some file, e.g., `rundirs.txt`.  Then run a Bash function to create the parameters files for all desired jobs, e.g.:

```bash
# Inputs
excel_columns_file="rundirs.txt"
fov_x="/home/weismanal/notebook/2019-02-11/inference/inference_images/fov_prepared-x_first.npy"
fov_y="/home/weismanal/notebook/2019-02-11/inference/inference_images/fov_prepared-y_first.npy"
fov_z="/home/weismanal/notebook/2019-02-11/inference/inference_images/fov_prepared-z_first.npy"
roi3_x="/home/weismanal/notebook/2019-02-11/inference/inference_images/roi3_prepared-x_first.npy"
roi3_y="/home/weismanal/notebook/2019-02-11/inference/inference_images/roi3_prepared-y_first.npy"
roi3_z="/home/weismanal/notebook/2019-02-11/inference/inference_images/roi3_prepared-z_first.npy"

IFS=$'\n'
for line in $(awk '{print $1, $2}' $excel_columns_file | awk -v FS=";" '{print $1}'); do

    # Obtain the true HP set name, final weights file, and last training parameters file for each set of hyperparameters
    hpset=$(echo $line | awk '{print $1}')
    dir=$(echo $line | awk '{print $2}')
    nmatches=$(find $dir -type f -iname "*.h5" | grep -v archive | wc -l)
    if [ $nmatches -gt 1 ]; then
        weightsfile=$(find $dir -type f -iname "*.h5" | grep -v archive | grep "hpset_$hpset")
    else
        weightsfile=$(find $dir -type f -iname "*.h5" | grep -v archive)
    fi
    outputdir=$(dirname $weightsfile)
    paramsfile=$(ls $outputdir/*.txt | grep -v "result.txt\|tmp.txt")
    
    # Call the function to create the inference parameters file from the last training parameters file
    make_inference_params_file $weightsfile $paramsfile $fov_x > params_files/params_${hpset}_fov_x.txt
    make_inference_params_file $weightsfile $paramsfile $fov_y > params_files/params_${hpset}_fov_y.txt
    make_inference_params_file $weightsfile $paramsfile $fov_z > params_files/params_${hpset}_fov_z.txt
    make_inference_params_file $weightsfile $paramsfile $roi3_x > params_files/params_${hpset}_roi3_x.txt
    make_inference_params_file $weightsfile $paramsfile $roi3_y > params_files/params_${hpset}_roi3_y.txt
    make_inference_params_file $weightsfile $paramsfile $roi3_z > params_files/params_${hpset}_roi3_z.txt

done
```

## (3) Set up the jobs

Run, e.g.:

```bash
FNLCR="/home/weismanal/checkouts/fnlcr-bids-hpc"
cd params_files
ijob=0
for file in *; do
    ijob=$[ijob+1]
    ijob2=$(printf '%03i' $ijob)
    jobdir="job_${ijob2}"
    mkdir ../inference_jobs/$jobdir
    pushd ../inference_jobs/$jobdir
    mv ../../params_files/$file .
    cp $FNLCR/candle/run_without_candle-template.sh run_without_candle.sh
    echo $file | grep resnet && model="resnet" || model="unet"
    awk -v jobname=$jobdir -v paramsfile=$(pwd)/$file -v model="${model}.py" '{gsub("12:00:00","00:45:00"); gsub("hpset_23",jobname); gsub("/home/weismanal/notebook/2019-01-28/jobs/not_candle/single_param_set.txt",paramsfile); gsub("unet.py",model); print}' $FNLCR/candle/run_without_candle-template.sh > run_without_candle.sh
    popd
done
cd ..
rmdir params_files
```

## (4) Run the jobs

Once you've confirmed that everything in the `inference_jobs` directory looks reasonable, you can submit the jobs, e.g.:

```bash
for jobdir in $(ls inference_jobs/); do
    pushd inference_jobs/$jobdir
    sbatch run_without_candle.sh
    popd
done
```

## Notes

* There is no need to worry about the scaling of the input images; they are scaled automatically using the normalize_images() Python function.
* Both the U-Net and ResNet Python scripts expect the labels to be in the range [0,1].