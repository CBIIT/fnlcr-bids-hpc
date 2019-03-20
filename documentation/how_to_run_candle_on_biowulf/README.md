# How to run CANDLE on Biowulf

## Step 1: Copy the submission script to a working directory

For example, assuming your working directory is `/home/`whomami/candle-test`, run this on Biowulf:

```bash
cd /home/`whomami`/candle-test
cp /data/BIDS-HPC/public/candle/Supervisor/templates/submit_candle_job.sh .
```

## Step 2: Optionally modify the variables in the submission script

Modify the submission script `submit_candle_job.sh` as desired, or leave it unmodified for an example run on the MNIST dataset.

## Step 3: Run the job

Submit the job by running, from the working directory,

```bash
./submit_candle_job.sh
```

(No, there really is no need for "sbatch".)

## Notes

* See the README [here](https://github.com/ECP-CANDLE/Supervisor/tree/develop/templates/README.md) for more details.
