~ [fnlcr-bids-hpc](https://cbiit.github.io/fnlcr-bids-hpc)
  ~ [documentation](https://cbiit.github.io/fnlcr-bids-hpc/documentation)
    ~ [How to create a central Git repository](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_create_a_central_git_repo)
    ~ [How to make your code CANDLE-compliant](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_make_your_code_candle_compliant)
    ~ **How to run CANDLE on Biowulf**
  ~ [How to use the image_segmentation module](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation)
    ~ [packages](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/packages)
      ~ [image_segmentation](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/packages/image_segmentation)
    ~ [sample_data](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/sample_data)
  ~ [modules](https://cbiit.github.io/fnlcr-bids-hpc/modules)
    ~ [candle](https://cbiit.github.io/fnlcr-bids-hpc/modules/candle)
  ~ [native_candle_build_on_biowulf](https://cbiit.github.io/fnlcr-bids-hpc/native_candle_build_on_biowulf)
    ~ [build_on_2019-04-04](https://cbiit.github.io/fnlcr-bids-hpc/native_candle_build_on_biowulf/build_on_2019-04-04)
  ~ [scripts](https://cbiit.github.io/fnlcr-bids-hpc/scripts)


---

# How to run CANDLE on Biowulf

## Step (1): Set up the environment

For example, assuming your working directory is `/data/smithja/candle-test`, run this on the Biowulf command line:

```bash
cd /data/smithja/candle-test
module load candle
```

## Step (2): Copy the submission script to a working directory

Run on Biowulf

```bash
copy_candle_template
```

in order to copy a template CANDLE submission script `submit_candle_job.sh` to, and create an empty `experiments` directory in, the working directory.

## Step (3) (optional): Modify the variables in the submission script

Modify the submission script `submit_candle_job.sh` as desired using [this link](XXXX) as a guide, or leave it unmodified for an example run on the MNIST dataset.

## Step (4): Run the job

Submit the job by running

```bash
./submit_candle_job.sh
```

(No, there really is no need for "sbatch".)

## Notes

* See the README [here](XXXX) for more details on how to use and modify this CANDLE template submission script.
