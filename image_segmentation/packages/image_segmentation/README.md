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
IMAGE_SEG=/Users/weismanal/checkouts/fnlcr-bids-hpc/image_segmentation/packages
```

#### Install the imgaug Python package from GitHub

See [here](https://imgaug.readthedocs.io/en/latest/source/installation.html) for instructions, i.e.,

```bash
pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
git clone https://github.com/aleju/imgaug.git
```

Note where this repository is on your filesystem, calling the path `IMGAUG`, e.g.,

```bash
IMGAUG=/Users/weismanal/checkouts/imgaug
```