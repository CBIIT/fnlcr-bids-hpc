#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --gres=gpu:k80:1
#SBATCH --time=12:00:00
#SBATCH --job-name=hpset_23

# Set up environment
module load python/3.6

# Set the file that the Python script below will read in order to determine the model parameters
export DEFAULT_PARAMS_FILE=/home/weismanal/notebook/2019-01-28/jobs/not_candle/single_param_set.txt

# Run the model
python /home/weismanal/checkouts/fnlcr-bids-hpc/models/unet.py
