#!/bin/bash

# This is based on instructions at https://github.com/ECP-CANDLE/Supervisor/tree/master/workflows as of 1/18/19.
# Compilation/testing done on compute nodes using command "sinteractive -n 3 -N 3 --gres=gpu:k20x:1 --mem=20G --no-gres-shell"
# --> actually now it's done using "sinteractive -n 2 -N 2 --gres=gpu:k20x:1 --no-gres-shell"

# These two variables (SWIFT_T_INSTALL and R_INSTALL) are needed... just create the new, empty directories here, and once everything is done installing, set to them the symbolic links /data/BIDS-HPC/public/software/builds/swift-t-install and /data/BIDS-HPC/public/software/builds/R.  Then test the build.

# For 4-4-19 build
#CANDLE=/data/BIDS-HPC/public/software/distributions/candle/2019-04-04
#export SWIFT_T_INSTALL=/data/BIDS-HPC/public/software/builds/versions/swift-t-install/swift-t-install-2019-04-04
#export R_INSTALL=/data/BIDS-HPC/public/software/builds/versions/R/R-2019-04-04
#module load R/3.5.0 gcc/7.3.0 openmpi/3.1.2/cuda-9.0/gcc-7.3.0-pmi2 tcl_tk/8.6.8_gcc-7.2.0 python/3.6 ant/1.10.3 java/1.8.0_181

# For 5-6-19 build
# CANDLE=/data/BIDS-HPC/public/software/distributions/candle/2019-05-06
# export SWIFT_T_INSTALL=$CANDLE/builds/swift-t-install
# export R_INSTALL=$CANDLE/builds/R
# module load R/3.5.0 tcl_tk/8.6.8_gcc-7.2.0 python/3.6 ant/1.10.3 java/1.8.0_181
# module remove openmpi/3.0.2/gcc-7.3.0
# module load gcc/7.2.0
# export LD_LIBRARY_PATH=/usr/local/slurm/lib:$LD_LIBRARY_PATH
# # # To add to PATH for execution:
# export PATH=/data/BIDS-HPC/public/software/builds/mpich-3.3-3/bin:$PATH
# # # To add for compilation against this build:
# export LD_LIBRARY_PATH=/data/BIDS-HPC/public/software/builds/mpich-3.3-3/lib:$LD_LIBRARY_PATH
# export LIBDIR=/data/BIDS-HPC/public/software/builds/mpich-3.3-3/lib:$LIBDIR
# export CPATH=/data/BIDS-HPC/public/software/builds/mpich-3.3-3/include:$CPATH
# export LD_PRELOAD=/usr/local/slurm/lib/libslurm.so:$LD_PRELOAD

module load mpich2/3.2.1/gcc-5.5.0 gcc/5.5.0

# #module load R/3.5.0 gcc/5.5.0 mpich2/3.2.1/gcc-5.5.0 tcl_tk/8.6.8_gcc-5.5.0 python/3.6 ant/1.10.3 java/1.8.0_181
# module load R/3.5.0
# module unload gcc/7.2.0
# module load mpich2/3.2.1/gcc-5.5.0 tcl_tk/8.6.8_gcc-5.5.0 python/3.6 ant/1.10.3 java/1.8.0_181
# module load gcc/5.5.0
# #MANPATH=/usr/local/GCC/5.5.0/share/man:/usr/local/Tcl_Tk/8.6.8/gcc_5.5.0/man:/usr/local/MPICH/3.2.1/gcc-5.5.0/share/man:/usr/local/apps/pandoc/2.1.1/share/man:/usr/local/OpenMPI/3.0.2/gcc-7.3.0/share/man:/usr/local/GSL/gcc-7.2.0/2.4/share/man:/usr/local/slurm/share/man:/usr/local/lmod/lmod/lmod/share/man::
# #LIBRARY_PATH=/usr/local/MPICH/3.2.1/gcc-5.5.0/lib:/usr/local/OpenMPI/3.0.2/gcc-7.3.0/lib/openmpi:/usr/local/OpenMPI/3.0.2/gcc-7.3.0/lib:/usr/local/GSL/gcc-7.2.0/2.4/lib:/usr/local/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64
# export LIBRARY_PATH=/usr/local/MPICH/3.2.1/gcc-5.5.0/lib:/usr/local/GSL/gcc-7.2.0/2.4/lib:/usr/local/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64
# #LD_LIBRARY_PATH=/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0/plugin:/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0/32:/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0:/usr/local/GCC/5.5.0/lib64:/usr/local/GCC/5.5.0/lib:/usr/local/Java/jdk1.8.0_181/lib:/usr/local/Tcl_Tk/8.6.8/gcc_5.5.0/lib:/usr/local/MPICH/3.2.1/gcc-5.5.0/lib:/usr/local/apps/geos/3.6.2/lib:/usr/local/ImageMagick/7.0.8/lib:/usr/local/OpenMPI/3.0.2/gcc-7.3.0/lib/openmpi:/usr/local/OpenMPI/3.0.2/gcc-7.3.0/lib:/usr/local/GSL/gcc-7.2.0/2.4/lib:/usr/local/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64
# export LD_LIBRARY_PATH=/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0/plugin:/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0/32:/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0:/usr/local/GCC/5.5.0/lib64:/usr/local/GCC/5.5.0/lib:/usr/local/Java/jdk1.8.0_181/lib:/usr/local/Tcl_Tk/8.6.8/gcc_5.5.0/lib:/usr/local/MPICH/3.2.1/gcc-5.5.0/lib:/usr/local/apps/geos/3.6.2/lib:/usr/local/ImageMagick/7.0.8/lib:/usr/local/GSL/gcc-7.2.0/2.4/lib:/usr/local/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64
# #CPATH=/usr/local/apps/geos/3.6.2/include:/usr/local/OpenMPI/3.0.2/gcc-7.3.0/include:/usr/local/GSL/gcc-7.2.0/2.4/include
# export CPATH=/usr/local/apps/geos/3.6.2/include:/usr/local/GSL/gcc-7.2.0/2.4/include
# #PATH=/usr/local/GCC/5.5.0/bin:/usr/local/Java/jdk1.8.0_181/bin:/usr/local/apps/ant/1.10.3/bin:/usr/local/Anaconda/envs/py3.6/bin:/usr/local/Tcl_Tk/8.6.8/gcc_5.5.0/bin:/usr/local/MPICH/3.2.1/gcc-5.5.0/bin:/usr/local/apps/pandoc/2.1.1/bin:/usr/local/apps/geos/3.6.2/bin:/usr/local/ImageMagick/7.0.8/bin:/usr/local/OpenMPI/3.0.2/gcc-7.3.0/bin:/usr/local/GSL/gcc-7.2.0/2.4/bin:/usr/local/apps/R/3.5/3.5.0_build2/bin:/usr/local/slurm/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/X11R6/bin:/usr/local/jdk/bin:/usr/bin:/usr/local/mysql/bin:/home/weismanal/bin:/home/weismanal/checkouts/fnlcr-bids-hpc/scripts:/opt/ibutils/bin
# export PATH=/usr/local/GCC/5.5.0/bin:/usr/local/Java/jdk1.8.0_181/bin:/usr/local/apps/ant/1.10.3/bin:/usr/local/Anaconda/envs/py3.6/bin:/usr/local/Tcl_Tk/8.6.8/gcc_5.5.0/bin:/usr/local/MPICH/3.2.1/gcc-5.5.0/bin:/usr/local/apps/pandoc/2.1.1/bin:/usr/local/apps/geos/3.6.2/bin:/usr/local/ImageMagick/7.0.8/bin:/usr/local/GSL/gcc-7.2.0/2.4/bin:/usr/local/apps/R/3.5/3.5.0_build2/bin:/usr/local/slurm/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/X11R6/bin:/usr/local/jdk/bin:/usr/bin:/usr/local/mysql/bin:/home/weismanal/bin:/home/weismanal/checkouts/fnlcr-bids-hpc/scripts:/opt/ibutils/bin
# #LD_RUN_PATH=/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0/plugin:/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0/32:/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0:/usr/local/GCC/5.5.0/lib64:/usr/local/GCC/5.5.0/lib:/usr/local/MPICH/3.2.1/gcc-5.5.0/lib:/usr/local/apps/geos/3.6.2/lib:/usr/local/OpenMPI/3.0.2/gcc-7.3.0/lib/openmpi:/usr/local/OpenMPI/3.0.2/gcc-7.3.0/lib:/usr/local/GSL/gcc-7.2.0/2.4/lib
# export LD_RUN_PATH=/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0/plugin:/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0/32:/usr/local/GCC/5.5.0/lib/gcc/x86_64-redhat-linux/5.5.0:/usr/local/GCC/5.5.0/lib64:/usr/local/GCC/5.5.0/lib:/usr/local/MPICH/3.2.1/gcc-5.5.0/lib:/usr/local/apps/geos/3.6.2/lib:/usr/local/GSL/gcc-7.2.0/2.4/lib


# Set up environment
# mkdir $SWIFT_T_INSTALL $R_INSTALL
# #export R_LIBS=$CANDLE/R/libs/
# export R_LIBS=$R_INSTALL/libs/
# mkdir $R_LIBS

#### TEST MPI COMMUNICATIONS ####
if [ 1 -eq 1 ]; then
    #mpicc hello.c
    #srun -n 3 a.out
    #srun -n 3 /home/weismanal/checkouts/fnlcr-bids-hpc/native_candle_build_on_biowulf/a.out
    
    #module rm gcc
    mpicc hello2.c
    #mpirun -n 3 a.out
    #mpirun -n 3 /home/weismanal/checkouts/fnlcr-bids-hpc/native_candle_build_on_biowulf/a.out
    mpirun -n 3 ./a.out
fi
# At least with only openmpi module loaded, confirm that everything's working correctly (note per Wolfgang Resch's email srun must be used instead of mpirun or mpiexec):
#weismanal@cn0613:~/notebook/2019-01-17 $ mpicc hello.c
#weismanal@cn0613:~/notebook/2019-01-17 $ srun -n 3 a.out
#[snip]
#Hello from node cn0613, rank 0 / 3 (CUDA_VISIBLE_DEVICES=1)
#Hello from node cn0614, rank 1 / 3 (CUDA_VISIBLE_DEVICES=0)
#Hello from node cn0615, rank 2 / 3 (CUDA_VISIBLE_DEVICES=0)

#### BUILD ####
if [ 0 -eq 1 ]; then
    # (1) Update CANDLE software
    # pushd $CANDLE/Supervisor # ensure we're in the fnlcr branch and that we've merged from develop
    # git pull; popd
    # pushd $CANDLE/Candle # ensure we're in master
    # git pull; popd;
    # pushd $CANDLE/Benchmarks # ensure we're in release_01? fnlcr I believe!! (and ensure we've merged from develop)
    # git pull; popd

    # (2) Install the CANDLE R packages --> don't forget to create $R_INSTALL/libs first!!
    # Yes, it may appear that the wrong version of GCC is being used but so far we've found this is fine
    $CANDLE/Supervisor/workflows/common/R/install-candle.sh |& tee -a candle-r_installation_out_and_err.txt

    # (3) Update Swift/T
    # pushd $CANDLE/swift-t
    # git pull; popd

    # (4) Set up the Swift/T installation
    # Be careful to update swift-t-settings.sh if swift-t-settings.sh.template was updated during the "git pull"
    #mv $CANDLE/swift-t/dev/build/swift-t-settings.sh $CANDLE/swift-t/dev/build/swift-t-settings-orig.sh
    ln -s $(pwd)/swift-t-settings.sh $CANDLE/swift-t/dev/build/swift-t-settings.sh

    # (5) Build and install Swift/T
    export NICE_CMD=""
    $CANDLE/swift-t/dev/build/build-swift-t.sh -v |& tee -a swift-t_installation_out_and_err.txt

    # (6) Save the build environment
    env > final_build_environment.txt

    # (7) Build EQ-R per the instructions $CANDLE/Supervisor/workflows/common/ext/EQ-R/eqr/COMPILING.txt
    # Ensure settings.sh.template hasn't changed by being pulled by git, otherwise, be careful!
    pushd $CANDLE/Supervisor/workflows/common/ext/EQ-R/eqr
    source ~-/eqr_settings.sh #|& tee -a eqr_installation_out_and_err.txt --> LEAVE THIS COMMENTED OUT; OTHERWISE THE SOURCE COMMAND DOES NOT WORK!
    ./bootstrap |& tee -a eqr_installation_out_and_err.txt
    ./configure --prefix=$PWD/.. |& tee -a eqr_installation_out_and_err.txt
    make install |& tee -a eqr_installation_out_and_err.txt
    env > final_build_environment-after_eqr.txt
    popd
    mv ~-/eqr_installation_out_and_err.txt ~-/final_build_environment-after_eqr.txt .
    
fi


#### TEST ####
# Testing done on "sinteractive -n 3 -N 3 --gres=gpu:k20x:1 --mem=20G --no-gres-shell"

#export OMPI_MCA_btl_openib_if_exclude="mlx4_0:1"
export PATH=$PATH:$CANDLE/swift-t-install/stc/bin
export PATH=$PATH:$CANDLE/swift-t-install/turbine/bin
export TURBINE_LOG=1
export ADLB_DEBUG_RANKS=1
export ADLB_DEBUG_HOSTMAP=1

#    # Test MPI
#    srun -n 1 python $CANDLE/Benchmarks/Pilot1/P1B3/p1b3_baseline_keras2.py

if [ 0 -eq 1 ]; then

    # (2)
    mpicc hello.c
    srun -n 3 a.out

    # (3) Test Swift/T
    #swift-t -n 3 mytest2.swift
    swift-t -n 3 mytest2.swift

    # (4)
    swift-t -n 3 -r $(pwd) myextension.swift

fi