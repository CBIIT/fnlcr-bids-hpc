* [fnlcr-bids-hpc](https://cbiit.github.io/fnlcr-bids-hpc)
  * [documentation](https://cbiit.github.io/fnlcr-bids-hpc/documentation)
    * [How to create a central Git repository](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_create_a_central_git_repo)
    * **How to make your code CANDLE-compliant**
    * [How to run CANDLE on Biowulf](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_run_candle_on_biowulf)
  * [How to use the image_segmentation module](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation)
    * [packages](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/packages)
      * [image_segmentation](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/packages/image_segmentation)
    * [sample_data](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/sample_data)
  * [modules](https://cbiit.github.io/fnlcr-bids-hpc/modules)
    * [candle](https://cbiit.github.io/fnlcr-bids-hpc/modules/candle)
  * [native_candle_build_on_biowulf](https://cbiit.github.io/fnlcr-bids-hpc/native_candle_build_on_biowulf)
    * [build_on_2019-04-04](https://cbiit.github.io/fnlcr-bids-hpc/native_candle_build_on_biowulf/build_on_2019-04-04)
  * [scripts](https://cbiit.github.io/fnlcr-bids-hpc/scripts)


---

# How to make your code CANDLE-compliant

The assumption is that your code is in Python and runs as-is on Biowulf.  For help with this, [contact us](mailto:andrew.weisman@nih.gov).

The backend doesn't matter; we have tested both Keras and PyTorch.  When in doubt, use Keras.

## Step (1): Place your *entire* Python script in a `run` function

**Before:**

```python
# First line of your code
# Second line of your code
# ...
# Last line of your code
```

**After:**

```python
def run(gParameters):
    # First line of your code
    # Second line of your code
    # ...
    # Last line of your code
    return(history) # ensure you define history in your code; it is likely the return variable from model.fit()
```

Don't forget to even wrap your import statements in the `run` function and to indent your code.

## Step (2): Prepend the `initialize_parameters` function

**Before:**

```python
def run(gParameters):
    # First line of your code
    # Second line of your code
    # ...
```

**After:**

```python
def initialize_parameters():

    # Add the candle_keras library to the Python path
    import sys, os
    sys.path.append(os.getenv("CANDLE")+'/Candle/common')

    # Instantiate the Benchmark class
    # The values of the prog and desc parameters don't really matter
    import candle_keras as candle
    mymodel_common = candle.Benchmark(os.path.dirname(os.path.realpath(__file__)), os.getenv("DEFAULT_PARAMS_FILE"), 'keras', prog='myprogram', desc='My CANDLE example')

    # Read the parameters (in a dictionary format) pointed to by the environment variable DEFAULT_PARAMS_FILE
    gParameters = candle.initialize_parameters(mymodel_common)

    # Return this dictionary of parameters
    return(gParameters)

def run(gParameters):
    # First line of your code
    # Second line of your code
    # ...
```

## Step (3): Append a `main` function and allow the script to be called from the command line

**Before:**

```python
    # ...
    # Last line of your code
    return(history) # ensure you define history in your code; it is likely the return variable from model.fit()
```

**After:**

```python
    # ...
    # Last line of your code
    return(history) # ensure you define history in your code; it is likely the return variable from model.fit()

def main():
    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == '__main__':
    main()
```

## Step (4): Define the variables you would like to vary in a hyperparameter optimization

For example, if you define the variable `epochs` (which likely sets the number of epochs to run) in the second line of your code, do this:

**Before:**

```python
def run(gParameters):
    # First line of your code
    epochs = 10
    # ...
```

**After:**

```python
def run(gParameters):
    # First line of your code
    epochs = gParameters['epochs']
    # ...
```

Note that this only needs to be done once per variable, when it is first defined.

While the dictionary "keys" (e.g., the argument to gParameters above) need not match the names of the variables being assigned, they must match the names of the keywords defined in the default parameters file, described next.

## Step (5): Create a default parameters file

In order to initialize the parameters in any given run, you need a configuration file that sets their default values.  Here is an example such file (located at `/data/BIDS-HPC/public/candle/Supervisor/templates/model_params/mnist1.txt`):

```
[Global_Params]
epochs=20
batch_size=128
activation='relu'
optimizer='rmsprop'
```

Note that this file is to be pointed to by the `DEFAULT_PARAMS_FILE` variable in the submission script `submit_candle_job.sh` described [here](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_run_candle_on_biowulf).

## Testing your script

You should be able to run your CANDLE-compliant script outside of the CANDLE "supervisor" by running your script manually on Biowulf.  This is an excellent test to show whether you've made your code properly CANDLE-compliant.

For example, we would test that we've made our template MNIST example (called `mnist_mlp.py`) properly CANDLE-compliant by placing the following in a file called `run_without_candle.sh` and running `sbatch run_without_candle.sh` on Biowulf:

```bash
#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --gres=gpu:k80:1
#SBATCH --time=00:05:00
#SBATCH --job-name=mnist_test_no_candle

# Set up environment
module load python/3.6 candle

# Set the file that the Python script below will read in order to determine the model parameters
export DEFAULT_PARAMS_FILE="$CANDLE/Supervisor/templates/model_params/mnist1.txt"

# Run the model
python $CANDLE/Supervisor/templates/models/mnist/mnist_mlp.py
```

In fact, this exact file is present in our CANDLE templates at `/data/BIDS-HPC/public/candle/Supervisor/templates/run_without_candle.sh`.  Feel free to run that file as-is to help you see how things work (`sbatch /data/BIDS-HPC/public/candle/Supervisor/templates/run_without_candle.sh`) or copy it to your local directory and edit it to test how well you made your code CANDLE-compliant (only the last two non-commented-out lines truly need be modified).

## Notes

* Sample CANDLE-compliant scripts can be found [here](https://github.com/ECP-CANDLE/Supervisor/tree/develop/templates/models).
* More details on writing CANDLE-compliant code can be found [here](https://ecp-candle.github.io/Candle/html/tutorials/writing_candle_code.html).
* You can check GPU utilization on a compute node using `watch nvidia-smi` (use Ctrl+C to exit out of the `watch` command).
