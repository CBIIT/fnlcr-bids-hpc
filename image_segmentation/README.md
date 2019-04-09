~ [fnlcr-bids-hpc](https://cbiit.github.io/fnlcr-bids-hpc)
  ~ [documentation](https://cbiit.github.io/fnlcr-bids-hpc/documentation)
    ~ [How to create a central Git repository](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_create_a_central_git_repo)
    ~ [How to make your code CANDLE-compliant](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_make_your_code_candle_compliant)
    ~ [How to run CANDLE on Biowulf](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_run_candle_on_biowulf)
  ~ **How to use the image_segmentation module**
    ~ [packages](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/packages)
      ~ [image_segmentation](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/packages/image_segmentation)
    ~ [sample_data](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/sample_data)
  ~ [modules](https://cbiit.github.io/fnlcr-bids-hpc/modules)
    ~ [candle](https://cbiit.github.io/fnlcr-bids-hpc/modules/candle)
  ~ [native_candle_build_on_biowulf](https://cbiit.github.io/fnlcr-bids-hpc/native_candle_build_on_biowulf)
    ~ [build_on_2019-04-04](https://cbiit.github.io/fnlcr-bids-hpc/native_candle_build_on_biowulf/build_on_2019-04-04)
  ~ [scripts](https://cbiit.github.io/fnlcr-bids-hpc/scripts)


---

# How to use the image_segmentation module

Note: At the moment, only the image_augmentation.py function is well-tested; the rest are coming soon!

## How to use the image_augmentation function

### Step 1: Install the required software

#### Install this Python package (called image_segmentation) from the fnlcr-bids-hpc repository on GitHub

```bash
git clone https://github.com/CBIIT/fnlcr-bids-hpc.git
```

Note where this repository is on your filesystem, and in particular, the `image_segmentation/packages` subdirectory.  Call this path `IMAGE_SEG`, e.g.,

```bash
export IMAGE_SEG=/data/weismanal/checkouts/fnlcr-bids-hpc/image_segmentation/packages
```

If you already have this repository checked out, do a `git pull` to ensure you have the latest updates.

#### Install the imgaug Python package from GitHub

**Note:** If you are on Biowulf, just run `export IMGAUG=/data/BIDS-HPC/public/software/checkouts/imgaug` and skip the installation.

See [here](https://imgaug.readthedocs.io/en/latest/source/installation.html) for instructions, i.e.,

```bash
pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
git clone https://github.com/aleju/imgaug.git
```

Note where this repository is on your filesystem, calling the path `IMGAUG`, e.g.,

```bash
export IMGAUG=/data/BIDS-HPC/public/software/checkouts/imgaug
```

### Step 2: Run an example image augmentation

In some working directory, e.g., `/data/weismanal/notebook/2019-03-21/testing_image_augmentation`, run the following:

```bash
cp $IMAGE_SEG/image_segmentation/image_augmentation_example.py .
# If on Biowulf: module load python/3.6
python image_augmentation_example.py
```

This will run a series of examples to provide starting points for using the scripts.  The examples locate Lady's nose.
