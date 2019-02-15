function get_training_info() {
    # Obtain the final weights file and training parameters file for the give training hyperparameter set name and output directory
    hpset=$1
    dir=$2
    nmatches=$(find $dir -type f -iname "*.h5" | grep -v archive | wc -l)
    if [ $nmatches -gt 1 ]; then
        weightsfile=$(find $dir -type f -iname "*.h5" | grep -v archive | grep "hpset_$hpset")
    else
        weightsfile=$(find $dir -type f -iname "*.h5" | grep -v archive)
    fi
    outputdir=$(dirname $weightsfile)
    paramsfile=$(ls $outputdir/*.txt | grep -v "result.txt\|tmp.txt")
    echo $weightsfile $paramsfile
}

function make_all_inference_params_files() {

    # Parameter
    training_output_dirs=$1 # e.g., training_output_dirs.txt

    # For each set of hyperparameters...
    IFS=$'\n'
    for line in $(cat $training_output_dirs); do

        hpset=$(echo $line | awk '{print $1}')
        dir=$(echo $line | awk '{print $2}')
        training_info=$(get_training_info $hpset $dir)
        weightsfile=$(echo $training_info | awk '{print $1}')
        paramsfile=$(echo $training_info | awk '{print $2}')

        # For each image on which inference should be done...
        for npy_file in inference_images/*; do
            bn=$(basename $(basename $npy_file) .npy)
            roi=$(echo $bn | awk -v FS="_" '{print $1}')
            firstdir=$(echo $bn | rev | awk -v FS="-" '{print $1}' | rev | awk -v FS="_" '{print $1}')
            make_single_inference_params_file $weightsfile $paramsfile $npy_file > params_files/params_${hpset}_${roi}_${firstdir}.txt
        done
        
    done    
}

function make_single_inference_params_file() {
    # Create an inference parameters file for running inference
    weightsfile=$1
    paramsfile=$2
    inf_images=$3
    awk -v doprint=1 '{if($0~/^verbose=/)doprint=0; if(doprint)print}' $paramsfile | grep -v "^images *=\|^labels *=\|^initialize *=\|^predict *=\|^epochs *=\|^encoder *=\|^batch_size *=\|^obj_return *="
    echo -e "images = '$inf_images'\ninitialize = '$weightsfile'\npredict = True"
}

function set_up_jobs() {
    cd params_files
    ijob=0
    for file in *; do
        echo $file
        ijob=$[ijob+1]
        ijob2=$(printf '%03i' $ijob)
        jobdir="job_${ijob2}"
        mkdir ../inference_jobs/$jobdir
        pushd ../inference_jobs/$jobdir > /dev/null
        mv ../../params_files/$file .
        echo $file | grep resnet > /dev/null && model="resnet" || model="unet"
        awk -v jobname=$jobdir -v paramsfile=$(pwd)/$file -v model="${model}.py" '{gsub("12:00:00","00:45:00"); gsub("hpset_23",jobname); gsub("/home/weismanal/notebook/2019-01-28/jobs/not_candle/single_param_set.txt",paramsfile); gsub("unet.py",model); print}' $IMAGE_SEG/../candle/run_without_candle-template.sh > run_without_candle.sh
        popd > /dev/null
    done
    cd ..
    rmdir params_files
}