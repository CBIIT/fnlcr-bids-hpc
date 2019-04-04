* [Home](https://cbiit.github.io/fnlcr-bids-hpc)
  * **Image segmentation**
  * [Documentation](https://cbiit.github.io/fnlcr-bids-hpc/documentation)
    * [How to create a central git repository](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_create_a_central_git_repo)


---

# Post-processing

## metrics_and_plots-driver.sh

Call this script in order to generate metrics and create plots for evaluating how well your model performs a 2D image segmentation task on particular regions of interest (ROIs) with known masks having two classes: background (0) and foreground (1).  All data should have the dimensions (N,H,W), where N is the number of Height-by-Width images in the stack.

In this documentation we assume the data are 3D so that N, H, and W actually correspond to what we define to be the z, y, and x directions, respectively, and three inferences must therefore be run---one on each direction (spatially correlated images; `<NINFERENCES>`=3).  However, the N axis can instead correspond to the stack dimension of completely uncorrelated images, in which case inference along only the N axis makes sense (spatially uncorrelated images; `<NINFERENCES>`=1).  In this case the only .npy files below should have the suffix "-z_first.npy"; there should not be any with the suffixes "-x_first.npy" or "-y_first.npy".

### Setup

1. Check out the current repository and set the variable IMAGE_SEG to where the image_segmentation directory is located, e.g.,

   `IMAGE_SEG=/Users/weismanal/checkouts/fnlcr-bids-hpc/image_segmentation`
2. In some working directory, place .npy files (or symbolic links to them) containing the images and true masks for the ROIs in the format `roiX_input_img.npy` and `known_masks_roiX.npy`, where X is an integer (1, 2, 3, etc.).
3. In the same working directory, create a directory with the model name for each model containing inference results.
4. In each model directory, place .npy files (or symbolic links to them) containing the inferred masks on the ROIs in the format `inferred_masks-roiX-Y_first.npy`, where X corresponds to the ROI on which inference was done and Y takes on the values x, y, or z and refers to the direction orthogonal to which the 2D inferences were performed.

   For example, if inference on the i-th image in the NxHxW stack is specified by img(i,:,:), then Y would be set to z due to the correspondence (N,H,W) <--> (z,y,x).  Similarly, Y would be set to y if inference were done on the j-th image specified by img(:,j,:), and Y would be set to x if inference were done on the k-th image specified by img(:,:,k).  As noted above, if the images in the stack are not correlated in space and only a single inference is done, Y should always be set to z for the single inference file.

*Explicit example of directory structure for spatially correlated images:*

```
├── 01-roi1_only_uncombined_unet
│   ├── inferred_masks-roi1-x_first.npy
│   ├── inferred_masks-roi1-y_first.npy
│   ├── inferred_masks-roi1-z_first.npy
│   ├── inferred_masks-roi2-x_first.npy
│   ├── inferred_masks-roi2-y_first.npy
│   ├── inferred_masks-roi2-z_first.npy
│   ├── inferred_masks-roi3-x_first.npy
│   ├── inferred_masks-roi3-y_first.npy
│   └── inferred_masks-roi3-z_first.npy
├── 02-roi2_only_uncombined_unet
│   ├── inferred_masks-roi1-x_first.npy
│   ├── inferred_masks-roi1-y_first.npy
│   ├── inferred_masks-roi1-z_first.npy
│   ├── inferred_masks-roi2-x_first.npy
│   ├── inferred_masks-roi2-y_first.npy
│   ├── inferred_masks-roi2-z_first.npy
│   ├── inferred_masks-roi3-x_first.npy
│   ├── inferred_masks-roi3-y_first.npy
│   └── inferred_masks-roi3-z_first.npy
├── 03-roi1+roi2_uncombined_unet
│   ├── inferred_masks-roi1-x_first.npy
│   ├── inferred_masks-roi1-y_first.npy
│   ├── inferred_masks-roi1-z_first.npy
│   ├── inferred_masks-roi2-x_first.npy
│   ├── inferred_masks-roi2-y_first.npy
│   ├── inferred_masks-roi2-z_first.npy
│   ├── inferred_masks-roi3-x_first.npy
│   ├── inferred_masks-roi3-y_first.npy
│   └── inferred_masks-roi3-z_first.npy
├── 04-roi1+roi2_combined_unet
│   ├── inferred_masks-roi1-x_first.npy
│   ├── inferred_masks-roi1-y_first.npy
│   ├── inferred_masks-roi1-z_first.npy
│   ├── inferred_masks-roi2-x_first.npy
│   ├── inferred_masks-roi2-y_first.npy
│   ├── inferred_masks-roi2-z_first.npy
│   ├── inferred_masks-roi3-x_first.npy
│   ├── inferred_masks-roi3-y_first.npy
│   └── inferred_masks-roi3-z_first.npy
├── 05-roi1+roi2_combined_resnet
│   ├── inferred_masks-roi1-x_first.npy
│   ├── inferred_masks-roi1-y_first.npy
│   ├── inferred_masks-roi1-z_first.npy
│   ├── inferred_masks-roi2-x_first.npy
│   ├── inferred_masks-roi2-y_first.npy
│   ├── inferred_masks-roi2-z_first.npy
│   ├── inferred_masks-roi3-x_first.npy
│   ├── inferred_masks-roi3-y_first.npy
│   └── inferred_masks-roi3-z_first.npy
├── known_masks_roi1.npy
├── known_masks_roi2.npy
├── known_masks_roi3.npy
├── roi1_input_img.npy
├── roi2_input_img.npy
├── roi3_input_img.npy
```

*Explicit example of directory structure for spatially uncorrelated images:*

```
├── 01-roi1_only_uncombined_unet
│   ├── inferred_masks-roi1-z_first.npy
│   ├── inferred_masks-roi2-z_first.npy
│   └── inferred_masks-roi3-z_first.npy
├── 02-roi2_only_uncombined_unet
│   ├── inferred_masks-roi1-z_first.npy
│   ├── inferred_masks-roi2-z_first.npy
│   └── inferred_masks-roi3-z_first.npy
├── 03-roi1+roi2_uncombined_unet
│   ├── inferred_masks-roi1-z_first.npy
│   ├── inferred_masks-roi2-z_first.npy
│   └── inferred_masks-roi3-z_first.npy
├── 04-roi1+roi2_combined_unet
│   ├── inferred_masks-roi1-z_first.npy
│   ├── inferred_masks-roi2-z_first.npy
│   └── inferred_masks-roi3-z_first.npy
├── 05-roi1+roi2_combined_resnet
│   ├── inferred_masks-roi1-z_first.npy
│   ├── inferred_masks-roi2-z_first.npy
│   └── inferred_masks-roi3-z_first.npy
├── known_masks_roi1.npy
├── known_masks_roi2.npy
├── known_masks_roi3.npy
├── roi1_input_img.npy
├── roi2_input_img.npy
├── roi3_input_img.npy
```

### Call format
 
```
$IMAGE_SEG/metrics_and_plots-driver.sh <MODELS> <ROI-NUMBERS> <NINFERENCES> <CALCULATE-METRICS> <CREATE-PLOTS> <MOVIE-DIRECTORY> <NFRAMES> <FRAMERATE>
```

where

1. `<MODELS>` is a whitespace-separated list of model names, i.e., the names of the directories of the models you created containing the inference .npy files

   * E.g., `"01-roi1_only_uncombined_unet 03-roi1+roi2_uncombined_unet"` or `$(find . -type d -iname "0*" | awk -v FS="./" '{print $2}' | sort)`
2. `<ROI-NUMBERS>` is a Python-formatted array of the ROI numbers on which you ran the inferences, e.g., `[1,2,3]`
3. `<NINFERENCES>` is the number of inferences you ran on each ROI using each model (this should be either `3` or `1`)
4. `<CALCULATE-METRICS>` is `1` if you want to calculate metrics in order to quantify how well the model performed, or `0` if you don't
5. `<CREATE-PLOTS>` is `1` if you want to create plots showing 2D images with overlays of the known and predicted  masks, or `0` if you don't
6. `<MOVIE-DIRECTORY>` is the name of the directory in which you want to place movies of the plots throughout the stacks, e.g., `movies`

   * Note: This directory will be created if it does not exist
7. `<NFRAMES>` is the number of frames in each movie (shouldn't be larger than the stack size), e.g., `40`
8. `<FRAMERATE>` is the framerate of the movies in frames per second, e.g., `2`

#### Sample calls

```bash
# Process the data but don't output anything
$IMAGE_SEG/metrics_and_plots-driver.sh "01-roi1_only_uncombined_unet 03-roi1+roi2_uncombined_unet" [1,2,3] 3 0 0 "" "" ""

# Output the metrics in 3d_metrics.txt
$IMAGE_SEG/metrics_and_plots-driver.sh "01-roi1_only_uncombined_unet 03-roi1+roi2_uncombined_unet" [1,2,3] 3 1 0 "" "" ""

# Create plots/movies but don't output metrics
$IMAGE_SEG/metrics_and_plots-driver.sh "01-roi1_only_uncombined_unet 03-roi1+roi2_uncombined_unet" [1,2,3] 3 0 1 /Users/weismanal/notebook/2018-12-12/movies 40 2

# Do both
$IMAGE_SEG/metrics_and_plots-driver.sh "01-roi1_only_uncombined_unet 03-roi1+roi2_uncombined_unet" [1,2,3] 3 1 1 /Users/weismanal/notebook/2018-12-12/movies 40 2

# Do both but for an uncorrelated image stack
$IMAGE_SEG/metrics_and_plots-driver.sh "01-roi1_only_uncombined_unet 03-roi1+roi2_uncombined_unet" [1,2,3] 1 1 1 /Users/weismanal/notebook/2018-12-12/movies 40 2
```

## metrics_and_plots.py

This file is called by metrics_and_plots-driver.sh and need not be called explicitly.
