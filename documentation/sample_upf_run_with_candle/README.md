# Step 1: Copy the submission script to a working directory

For example, assuming your working directory is `/home/weismanal/notebook/2019-03-18/test3`, run this on Biowulf:

```bash
cd /home/weismanal/notebook/2019-03-18/test3
cp /data/BIDS-HPC/public/candle/Supervisor/templates/submit_candle_job.sh .
```

# Step 2: Modify the variables in the submission script

Set the values of the variables in the submission script as follows:

```bash
export EXPERIMENTS="/home/weismanal/notebook/2019-03-18/test3" # the results of the UPF job will be placed here
export MODEL_NAME="mnist_upf_test" # descriptive name for the job
export OBJ_RETURN="val_loss" # objective function to extract
export PROCS="4" # PROCS-2 are actually used for computation
export WALLTIME="00:30:00" # set the job to run for 30 minutes at most
export GPU_TYPE="k80" # run each job on an NVIDIA K80 GPU
export MODEL_PYTHON_DIR="$CANDLE_DIR/Supervisor/templates/models/mnist" # directory containing the model (note $CANDLE_DIR is already set for Biowulf)
export MODEL_PYTHON_SCRIPT="mnist_mlp" # the model itself (without the ".py" extension)
export DEFAULT_PARAMS_FILE="$CANDLE_DIR/Supervisor/templates/model_params/mnist1.txt" # set the default parameters for the model
export WORKFLOW_TYPE="upf" # run the unrolled parameter file workflow in CANDLE (this is the default)
export WORKFLOW_SETTINGS_FILE="$CANDLE_DIR/Supervisor/templates/workflow_settings/upf3.txt" # specify the unrolled parameters file
```

# Step 3: Run the job

Submit the job by running, from the working directory,

```bash
./submit_candle_job.sh
```

# Notes

See the README [here](https://github.com/ECP-CANDLE/Supervisor/tree/develop/templates/README.md) for more details.