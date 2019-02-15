# Pre-inference workflow

**Prerequisites:** It is assumed you have completed training all models.

## (0) Set-up

Clone the fnlcr-bids-hpc repository, load the latest Python installation on Biowulf, set up the Python and Bash paths to include the image_segmentation libraries, and create the empty required directories, e.g.,

```bash
# In Bash:
git clone git@github.com:CBIIT/fnlcr-bids-hpc.git /home/weismanal/checkouts
module load python/3.6
export PYTHONPATH=$PYTHONPATH:/home/weismanal/checkouts/fnlcr-bids-hpc/image_segmentation
export IMAGE_SEG=/home/weismanal/checkouts/fnlcr-bids-hpc/image_segmentation
mkdir inference_images params_files inference_jobs
```

## (1) Create preprocessed inference images

Preprocess the inference images by scaling them to the format required for the model, exposing the other two dimensions of the images, and padding the 2D planes of the resulting three images so they can be run through an `nlayers_max`-layer U-Net:

```python
# In Python:
from image_segmentation import preprocess_inference_images
preprocess_inference_images(images_npy_file, idtype, nlayers_max, prefix)
```

For example,

```python
# In Python:
from image_segmentation import preprocess_inference_images
roi3_raw = '/home/weismanal/links/1-pre-processing/roi3/1-not_padded/roi3_images_original.npy'
fov_raw = '/home/weismanal/links/1-pre-processing/field_of_view/field_of_view.npy'
preprocess_inference_images(roi3_raw, 2, 6, 'inference_images/roi3_prepared')
preprocess_inference_images(fov_raw, 2, 6, 'inference_images/fov_prepared')
```

will produce six images in the inference_images directory:

```
roi3_prepared-x_first.npy
roi3_prepared-y_first.npy
roi3_prepared-z_first.npy
fov_prepared-x_first.npy
fov_prepared-y_first.npy
fov_prepared-z_first.npy
```

The images will have been scaled to uint16 (from `idtype`=2; this is the format expected by both the U-Net and ResNet scripts) and padded so they can be run through a six-layer U-Net (from `nlayers_max`=6).  Note that there is no need to worry about the scaling of the input images; they are scaled automatically using the normalize_images() Python function.

## (2) Create the inference parameters files

Create a text file containing two whitespace-separate columns: (1) some text identifier of the model's hyperparameter set and (2) the corresponding directory containing the training results, e.g.,

```
10 /home/weismanal/notebook/2019-01-20/upf_kedar_full/experiments/X010
11 /home/weismanal/notebook/2019-01-28/jobs/candle/experiments/X003
16 /home/weismanal/notebook/2019-01-20/upf_kedar_full/experiments/X010
17 /home/weismanal/notebook/2019-01-20/upf_kedar_full/experiments/X010
22 /home/weismanal/notebook/2019-01-20/upf_kedar_full/experiments/X010
23 /home/weismanal/notebook/2019-01-28/jobs/candle/experiments/X003
28 /home/weismanal/notebook/2019-01-20/upf_kedar_full/experiments/X010
30 /home/weismanal/notebook/2019-02-11/unet_on_kedars_data-hpset_30
32 /home/weismanal/notebook/2019-02-07/more_complex_models_with_batch_norm/hpset_32/continued/continued
33 /home/weismanal/notebook/2019-02-07/more_complex_models_with_batch_norm/hpset_33/continued/continued
34 /home/weismanal/notebook/2019-01-28/jobs/candle/experiments/X003
21a /home/weismanal/notebook/2019-01-28/jobs/candle/experiments/X004
21b /home/weismanal/notebook/2019-01-28/jobs/candle/experiments/X004
21c /home/weismanal/notebook/2019-01-28/jobs/candle/experiments/X004
21d /home/weismanal/notebook/2019-01-28/jobs/candle/experiments/X004
last_good_unet /home/weismanal/notebook/2019-01-28/jobs/not_candle/continuation
resnet /home/weismanal/notebook/2019-02-11/resnet_on_kedars_data
```

Then create the input parameters files for running inference using Keras:

```bash
# In Bash:
. $IMAGE_SEG/image_segmentation.sh
make_all_inference_params_files <training_output_dirs_file>
```

For example,

```bash
# In Bash:
. $IMAGE_SEG/image_segmentation.sh
make_all_inference_params_files training_output_dirs.txt
```

will produce NHPSETS x NINFIMAGES parameters files in the params_files directory, where NHPSETS is the number of hyperparameter sets (i.e., lines in training_output_dirs.txt) and NINFIMAGES is the number of .npy files in the inference_images directory on which inference will be run.

## (3) Create and organize all files needed for running the jobs

Move the parameters files to and create and modify the job submission template in the inference_jobs directory by running

```bash
# In Bash:
. $IMAGE_SEG/image_segmentation.sh
set_up_jobs
```

## (4) Submit the jobs to the queue

Once you've confirmed that everything in the inference_jobs directory looks reasonable, you can submit the jobs by, e.g.,

```bash
# In Bash
for jobdir in $(ls inference_jobs/); do
    pushd inference_jobs/$jobdir > /dev/null
    sbatch run_without_candle.sh
    popd > /dev/null
done
```

# Post-processing workflow

