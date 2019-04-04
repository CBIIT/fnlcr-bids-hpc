# Inputs
excel_columns_file="rundirs.txt"
fov_x="/home/weismanal/notebook/2019-02-11/inference/inference_images/fov_prepared-x_first.npy"
fov_y="/home/weismanal/notebook/2019-02-11/inference/inference_images/fov_prepared-y_first.npy"
fov_z="/home/weismanal/notebook/2019-02-11/inference/inference_images/fov_prepared-z_first.npy"
roi3_x="/home/weismanal/notebook/2019-02-11/inference/inference_images/roi3_prepared-x_first.npy"
roi3_y="/home/weismanal/notebook/2019-02-11/inference/inference_images/roi3_prepared-y_first.npy"
roi3_z="/home/weismanal/notebook/2019-02-11/inference/inference_images/roi3_prepared-z_first.npy"

IFS=$'\n'
for line in $(awk '{print $1, $2}' $excel_columns_file | awk -v FS=";" '{print $1}'); do

    # Obtain the true HP set name, final weights file, and last training parameters file for each set of hyperparameters
    hpset=$(echo $line | awk '{print $1}')
    dir=$(echo $line | awk '{print $2}')
    nmatches=$(find $dir -type f -iname "*.h5" | grep -v archive | wc -l)
    if [ $nmatches -gt 1 ]; then
        weightsfile=$(find $dir -type f -iname "*.h5" | grep -v archive | grep "hpset_$hpset")
    else
        weightsfile=$(find $dir -type f -iname "*.h5" | grep -v archive)
    fi
    outputdir=$(dirname $weightsfile)
    paramsfile=$(ls $outputdir/*.txt | grep -v "result.txt\|tmp.txt")
    
    # Call the function to create the inference parameters file from the last training parameters file
    make_inference_params_file $weightsfile $paramsfile $fov_x > params_files/params_${hpset}_fov_x.txt
    make_inference_params_file $weightsfile $paramsfile $fov_y > params_files/params_${hpset}_fov_y.txt
    make_inference_params_file $weightsfile $paramsfile $fov_z > params_files/params_${hpset}_fov_z.txt
    make_inference_params_file $weightsfile $paramsfile $roi3_x > params_files/params_${hpset}_roi3_x.txt
    make_inference_params_file $weightsfile $paramsfile $roi3_y > params_files/params_${hpset}_roi3_y.txt
    make_inference_params_file $weightsfile $paramsfile $roi3_z > params_files/params_${hpset}_roi3_z.txt

done