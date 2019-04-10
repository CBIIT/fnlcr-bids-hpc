#!/bin/bash

src_dir=$(cd ../../; pwd)
export PYTHONPATH=$PYTHONPATH:$src_dir

if [ -z  "$1" ]
then
    echo "need a python script to run" >&2
    echo "example: ./run-test.sh test.py" >&2
    exit
fi
python $1 
