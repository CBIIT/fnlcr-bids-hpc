---
bigimg: "/img/FNL_ATRF_Pano_4x10.jpg"
title: How to run CANDLE on Biowulf
---
This page will show you (1) how to run a sample job using CANDLE and (2) what needs to be modified in order to run your own job using CANDLE.  For more information on CANDLE, please click [here](XXXX).

## Step (1): Set up your environment

First, enter a working directory and load the CANDLE environment using `module load candle`.  For example, if your working directory is `/data/USERNAME/candle-test`, run this on the Biowulf command line:

```bash
cd /data/USERNAME/candle-test
module load candle
```

## Step (2): Copy the template submission script to the working directory

Copy the template CANDLE submission script `submit_candle_job.sh` to, and create an empty `experiments` directory in, the working directory by running on Biowulf:

```bash
copy_candle_template
```

## Step (3) (Optional): Modify the variables in the submission script

Modify the submission script `submit_candle_job.sh` as desired using [this link](XXXX) as a guide, or leave it unmodified for an example run on the MNIST dataset.

## Step (4): Run the job

Submit the job by running:

```bash
./submit_candle_job.sh
```

(No, there really is no need for "sbatch".)

## Notes

* See the README [here](XXXX) for more details on how to use and modify the CANDLE template submission script.
* Feel free to email [Andrew Weisman](mailto:andrew.weisman@nih.gov) with any questions.