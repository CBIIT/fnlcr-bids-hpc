#!/bin/bash

# Usage: ./submit_candle_job.sh
# Note: This script, the DEFAULT_PARAMS_FILE file, and the WORKFLOW_SETTINGS_FILE file should be the ONLY things to modify/set prior to running CANDLE jobs

# Constants
CANDLE=/data/BIDS-HPC/public/candle
SITE=biowulf
CFG_SYS=$CANDLE/Supervisor/workflows/common/sh/cfg-sys-${SITE}.sh
SCRIPTDIR=$(cd $(dirname $0); pwd) # obtain the directory in which this script (submit_candle_job.sh) lies in order to be optionally used in the settings below

################ EDIT PARAMETERS BELOW ################
# Scheduler settings
PROCS=2 # remember that PROCS-1 are actually used for DL jobs (for UPF...I think it's PROCS-2 for mlrMBO)
PPN=1
WALLTIME=12:00:00
GPU_TYPE=p100 # choices are p100, k80, v100, k20x
MEM_PER_NODE=20G # just moved down from 50G and may have to go back up if get memory issues!!!

# Model settings that aren't gParameters (they're processed before the MODEL_PYTHON_SCRIPT)
# Note that $obj_return is also processed prior to MODEL_PYTHON_SCRIPT for some reason, but it should definitely be a parameter in the model
WORKFLOW_TYPE=upf # upf, mlrMBO, etc.
WORKFLOW_SETTINGS_FILE=$SCRIPTDIR/upf.txt # e.g., the unrolled parameter file
EXPERIMENTS=$SCRIPTDIR/experiments
#MODEL_PYTHON_DIR=$CANDLE/Supervisor/workflows/examples
#MODEL_PYTHON_SCRIPT=unet
MODEL_PYTHON_DIR=$CANDLE/Benchmarks2/Pilot1/Uno
MODEL_PYTHON_SCRIPT=uno_baseline_keras2
MODEL_NAME="testing_uno-mytest"
DEFAULT_PARAMS_FILE="$SCRIPTDIR/default_params_file.txt" # It's necessary to export this variable so that it can be picked up by the MODEL_PYTHON_SCRIPT by using the full pathname. E.g., if we just used the filename default_params.txt hardcoded into the MODEL_PYTHON_SCRIPT, the script would look this global parameter file in the same directory that it's in (i.e., MODEL_PYTHON_DIR), but that would preclude using a MODEL_PYTHON_SCRIPT that's a symbolic link, i.e., we'd have to always copy the MODEL_PYTHON_SCRIPT to the current working directory, which is inefficient
################ EDIT PARAMETERS ABOVE ################

# Export variables needed later
#export EMEWS_PROJECT_ROOT=$CANDLE/Supervisor/workflows/$WORKFLOW_TYPE MODEL_NAME OBJ_RETURN=$obj_return EXPERIMENTS PROCS PPN WALLTIME MODEL_PYTHON_DIR MODEL_PYTHON_SCRIPT GPU_TYPE MEM_PER_NODE DEFAULT_PARAMS_FILE
export EMEWS_PROJECT_ROOT=$CANDLE/Supervisor/workflows/$WORKFLOW_TYPE MODEL_NAME OBJ_RETURN="val_loss" EXPERIMENTS PROCS PPN WALLTIME MODEL_PYTHON_DIR MODEL_PYTHON_SCRIPT GPU_TYPE MEM_PER_NODE DEFAULT_PARAMS_FILE

# Call the workflow
$EMEWS_PROJECT_ROOT/swift/workflow.sh $SITE -a $CFG_SYS $WORKFLOW_SETTINGS_FILE