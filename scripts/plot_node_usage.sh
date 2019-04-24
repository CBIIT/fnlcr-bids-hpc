#!/bin/bash

# Sample calls:
#
#   /data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc/scripts/plot_node_usage.sh "/home/weismanal/notebook/2019-04-23/testing_compute_node_usage/last-exp" "(10,4)" "."
#   /data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc/scripts/plot_node_usage.sh "$(pwd)/last-exp" "(10,4)" "last-exp"
#

# Process the inputs: get the location of the symbolic link pointing to the last job output and the new figure size and location
exp_dir=$1 # e.g., "/home/weismanal/notebook/2019-04-23/testing_compute_node_usage/last-exp"
figsize=$2 # e.g., "(10,4)"
figdir=$3 # e.g., "last-exp" or "."

# Name a temporary text file to hold the timing data
datafile="timing_data.txt"

# Output the timing data into the temporary text file
grep "^MODEL.SH START TIME: \|^HOST: \|^RUNID: \|^MODEL.SH END TIME: " $(find ${exp_dir}/ -name model.log | sort) | awk '{printf("%s ",$NF); if (NR%4==0) print ""}' > $datafile

# Run the Python script that processes the timing data
module load python/3.6
python /data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc/scripts/plot_node_usage.py $datafile $figsize $figdir

# Delete the temporary text file
rm -f $datafile