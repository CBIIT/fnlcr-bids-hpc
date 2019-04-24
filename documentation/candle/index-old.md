---
bigimg: "/img/FNL_ATRF_Pano_4x10.jpg"
title: Introduction to using CANDLE on Biowulf
---
## What is CANDLE?

CANDLE is a software package primarily used to perform the crucial step of hyperparameter optimization for developing deep learning models.  In a nutshell, when you are developing a deep learning model, a significant amount of computation time is spent fitting the "weights" or "parameters" to the model you have chosen.  But how do you choose a model in the first place?  You must first decide on the *type* of model you'd like to try (such as a neural network; choosing the *type* of model can be an art in its own right), and then you need to choose the best "settings" or "hyperparameters" that concretely define the model itself (such as the number of layers in a neural network).  The process of hyperparameter optimization helps you select the best set of hyperparameters to try, e.g., the set that produces returns the lowest loss value for a set number of iterations (or "epochs") of the algorithm that optimizes the *weights*.

There are a number of ways to perform hyperparameter optimization (HPO) on a given type of model.  The simplest is to define the space of hyperparameters and to then test out all possible combinations of hyperparameters within that space.  This is commonly referred to as the "grid search" algorithm, or, as it is called in CANDLE, the Unrolled Parameter File (UPF) method.  A more intelligent way for performing HPO is to use a Bayesian technique that utilizes information about how well a previous set of hyperparameters performed in choosing the next set of hyperparameters to try.  The Bayesian HPO algorithm used in CANDLE utilizes a package written in R called mlrMBO.  Additional HPO algorithms, such as those based on random searches and population-based training (PBT) are available in CANDLE.

The strength of CANDLE lies in the ease with which a user can perform HPO using various algorithms and in its scalability; CANDLE has successfully been run on top-ten supercomputers using thousands of nodes for a single HPO job.  It is a mature, well-supported code that has recently been installed natively on Biowulf in order to help lower the barrier for researchers at NIH to effectively develop robust deep learning models for their data by reducing the time required for performing HPO.  Supported deep learning backends include the popular Keras and PyTorch packages.

"All" that is required in order to use CANDLE is a model (implemented in a Python** script) that already runs on Biowulf and some idea of the hyperparameter space that you would like to explore.  Then you would make your script [CANDLE-compliant](XXXX) (a straightforward and quick process) and reference the hyperparameter space directly in the Biowulf submission script along with the HPO algorithm you would like to use.  Then you simply submit the script to Biowulf and let CANDLE efficiently determine the optimal set of hyperparameters to use for your model.  The next steps, which can be performed with or without CANDLE, would be relatively straightforward; armed with your model that's parametrized by an optimal set of hyperparameters, you can then train it as deeply as is reasonable (using an optimization algorithm) on your training set in order to choose the best set of weights/parameters for your data, after which you can simply "run" the model in "inference mode" to make predictions on new datasets.

The Internet is rich with more information on this process, and we're [here to help](mailto:andrew.weisman@nih.gov) you with any questions you might have along the way.

## So what's next?

The easiest way to start would be to [run an example HPO](XXXX) using CANDLE that works out-of-the-box on Biowulf!  It takes just a few minutes to run and demonstrates [how to run CANDLE on Biowulf](XXXX), [what needs to be done](XXXX) in order to run an HPO on your model using CANDLE on Biowulf, and how to [make your code CANDLE-compliant](XXXX).  And remember that we're [here to help](mailto:andrew.weisman@nih.gov) with any of these steps!

## **A note regarding language support in CANDLE

As Python is becoming the language of data science, CANDLE primarily supports code written in Python.  So if you are writing a deep learning model from scratch, or the one you already have is easy to convert to Python, we recommend you use Python.  It is a relatively intuitive scripting language and free online Python tutorials abound.

That said, we do have burgeoning support for scripts written in other languages such as R.  If this functionality is important to you, [we are interested in hearing from you](mailto:andrew.weisman@nih.gov), and we are working on more complete documentation for using [language-agnostic scripts](https://github.com/ECP-CANDLE/Supervisor/tree/develop/templates/language_agnostic) we have recently written to serve this purpose.

Please note that language of choice we are referring to is the one used to write the *main* script that would run without CANDLE.  For example, if you were to run your deep learning model using the command `python mymodel.py` the language you are using is Python, and if you ran your model using `R mymodel.R` the language you are using is R.  Main scripts such as these often call modules written in other languages.  For example, if your main script is written in Python and contains the code

```python
import os
os.subprocess('R do_some_work.R')
```

then even though R is *used* in the main script, the main script is still *written* in Python.  If yours is such a case, then as far as CANDLE is concerned *your script is written in Python*, and, e.g., you can follow the straightforward instructions [here](XXXX) for making your code CANDLE-compliant; there is no need to use the [language-agnostic scripts](XXXX).

Feel free to email [Andrew Weisman](mailto:andrew.weisman@nih.gov) with further questions about language support in CANDLE.