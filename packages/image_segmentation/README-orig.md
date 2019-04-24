# Pre-inference workflow

**Goal:** Prepare for -- and initiate -- running inference (i.e., the semantic segmentation task) on some images using models that you have completed training.

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
roi3_raw = '/data/weismanal/notebook/2018-12-11/CellB_R_IASEM.npy'
fov_raw = '/data/weismanal/originals/mammalian_sweep/data/images/padded_images_cellB_FOV_Mito.npy'
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

# Postprocessing workflow

**Goal:** Evaluate how well your models performed semantic segmentation on images with known masks by generating metrics and movies.

## (0) Set-up

Clone the fnlcr-bids-hpc repository, load the latest Python and FFmpeg installations on Biowulf, and set up the Python and Bash paths to include the image_segmentation libraries, e.g.,

```bash
# In Bash:
git clone git@github.com:CBIIT/fnlcr-bids-hpc.git /home/weismanal/checkouts
module load python/3.6 FFmpeg
export PYTHONPATH=$PYTHONPATH:/home/weismanal/checkouts/fnlcr-bids-hpc/image_segmentation
export IMAGE_SEG=/home/weismanal/checkouts/fnlcr-bids-hpc/image_segmentation
```

## (1) Set up the directory structure using symbolic links

Create symbolic links in the current directory to the images and corresponding known masks, e.g.,

```bash
# In Bash:
ln -s /data/weismanal/notebook/2018-12-11/CellB_R_IASEM.npy roi3_input_img.npy
ln -s /data/weismanal/notebook/2018-12-11/CellB_R_IASEM-label.npy known_masks_roi3.npy
```

While of course these files need not be linked to (they may simply be referenced later) nor have the filename format

```
roiX_input_img.npy
roiY_input_img.npy
...
known_masks_roiX.npy
known_masks_roiY.npy
...
```

for consistency with what follows it helps to adopt this naming convention and to place the links in the current working directory.

Since there are potentially many inferred masks, it's best to create the links automatically; if the naming conventions of the pre-inference workflow section are followed, this can be accomplished via

```bash
# In Bash:
. $IMAGE_SEG/image_segmentation.sh
create_links_to_inferred_masks <inference_results_directory>
```

For example,

```bash
# In Bash:
. $IMAGE_SEG/image_segmentation.sh
create_links_to_inferred_masks /home/weismanal/notebook/2019-02-11/inference/inference_jobs
```

will create NHPSETS numbered directories each containing links to all the masks inferred by the corresponding model, e.g.,

```
...
08-hpset_21d/inferred_masks-fov-x_first.npy
08-hpset_21d/inferred_masks-fov-y_first.npy
08-hpset_21d/inferred_masks-fov-z_first.npy
08-hpset_21d/inferred_masks-roi3-x_first.npy
08-hpset_21d/inferred_masks-roi3-y_first.npy
08-hpset_21d/inferred_masks-roi3-z_first.npy
09-hpset_22/inferred_masks-fov-x_first.npy
09-hpset_22/inferred_masks-fov-y_first.npy
09-hpset_22/inferred_masks-fov-z_first.npy
09-hpset_22/inferred_masks-roi3-x_first.npy
09-hpset_22/inferred_masks-roi3-y_first.npy
09-hpset_22/inferred_masks-roi3-z_first.npy
...
```

## (2) Load the data

Use the functions

```python
# In Python
load_images(npy_file)
load_inferred_masks(roi, unpadded_shape, models, inference_directions)
```

to load and preprocess the data for all further types of analysis (whether metric calculation or movie creation).  E.g.,

```python
# In Python
from image_segmentation import load_images, load_inferred_masks
roi = 'roi3'
models = ['08-hpset_21d','09-hpset_22']
inference_directions = ['x','y','z']
images = load_images(roi+'_input_img.npy')
known_masks = load_images('known_masks_'+roi+'.npy')
inferred_masks = load_inferred_masks(roi, images.shape, models, inference_directions)
```

will load the images in the `images` array, the known masks in the `known_masks` array, and the masks inferred by every specified model in every specified inference direction in the `inferred_masks` array.

```python
# In Python
from image_segmentation import calculate_metrics
metrics_2d, metrics_3d = calculate_metrics(known_masks,inferred_masks)
```