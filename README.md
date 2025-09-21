## Introduction

A very efficient python implementation and framework for training neural networks on CPUs.
Designed with class hierarchy and generic code allowing for modular implementation and wiring of various types of neural network layers.

It does not perform symbolic differentiation, but for the supported layers it implements the gradients from their analytical formulations, with algebraic optimizations and vectorization across the batch and sometimes the time dimension.
All gradient implementations are verified with gradient checks across all dimensions.

It provides standard feed-forward layer, standard and GRU recurrent layers (RNN), bi-directional RNN layer, embedding lookup layer, multi-class logistic (softmax) regression for classification, linear-chain Conditional Random Fields output layer for sequence modeling classification, squared-error loss output layer for regression.
It implements batching of sequences (of uneven lengths) for the RNN layer, embedding lookup layer and softmax layer.
Implementation for four variants of stochastic gradient descend are included, with additional features such as periodic reporting of training loss and periodic evaluation on a held-out data set.
It uses [NumPy](https://numpy.org/) for efficient matrix operations and matrix views.
For a small portion of the implementation where gains from it are possible, [Cython](https://cython.org/) is used.

This framework has been applied successfully to train competitive deep neural networks for the Named Entity Recognition (NER) task reproducing published paper results in 2017.


## Design

We consider multi-layer neural networks where the top layer is a scalar loss function. When training such networks for each layer we need expressions for the following two types of derivatives. Given the derivative of the loss function with respect to this layer's output variables (or equivalently the immediately upper layer input variables) we need to compute the:

1. Derivative of the loss function with respect to this layer's trainable parameters
2. Derivative of the loss function with respect to this layer's input variables 

The concatenation of (1) across all layers is the derivative of the full model and is used by the stochastic gradient descend algorithm to determine the next values of the trainable parameters of the full model.
Derivative (2) is necessary for computing (1): it allows back-propagating the error from the top-most scalar layer incrementally across each layer going to the bottom layer using the chain rule.

The general framework for supporting the above is in file [neural_layer.py](pyneural/neural_layer.py).
Auxiliary classes showing how to combine layers are in [layers.py](pyneural/layers.py).

The framework supports and requires the model and gradient of the full network to be each inside a contiguous memory buffer.
The nested components model and gradient are references to the appropriate places inside these memory buffers that are defined at the top-most enclosing network object.
These references are set recursively during wiring of the layers.
During the fitting procedure, the model and gradient buffers are updated strictly in-place and are *never* copied.
These architectural features are critical for a very fast training procedure.
The fitting procedure itself is inside [sgd.py](pyneural/sgd.py).

This framework favors minimizing execution time almost unconditionally over minimizing memory consumption and implementation complexity.
Any intermediate results that can cached to be used in back-propagation are cached instead of recomputed.
Vectorization is used heavily.
An unusual feature and conscious design choice is that the RNN implementations are *vectorized on the time dimension* as well as the batch dimension.
The minimum possible work is done inside the RNN loop, its results are accumulated and the final matrix operations are applied as single operations across all time steps.
This makes especially the backward step far more complex to implement but also much faster in a numpy-based implementation.


## Build Instructions

This project requires python 3.9 or newer, but later versions are preferable for more up-to-date dependent packages.
The few direct dependencies are listed in [requirements.txt](requirements.txt).
For dependency isolation it is strongly recommended to use a python virtual environment using [python3 virtualenv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/).
(A standard python virtualenv inherits the python interpreter of the environment it was created from, while conda allows you to install any python interpreter version you wish.)
You should use the [build](https://build.pypa.io/en/stable/) frontend instead of deprecated direct command line calls to `setup.py`.

After you create and activate the environment, install dependencies as following:
* for non-conda environment: `pip install -r requirements.txt`
* for conda environment: remove `build` from `requirements.txt` because it does not exist as conda package, then `mamba install --file requirements.txt`, `pip install build`.

Build the binary wheel distribution using `python -m build --wheel`, which correctly handles the cython code present in this project.
The `--wheel` switch compiles cython to C and C to native machine code.
For local development or for executing the tests, the compiled C code needs to be copied to the location where the local package is present.
The simplest way to achieve this is to install the project in "development mode".
Execute at the command line, from the top-level directory: `python -m pip install -e .`


## Testing and Correctness

An earlier version that differs in minor ways was extensively used on macOS (10.13.3 and similar) and ubuntu 16.04 to train competitive models for the Named Entity Recognition NLP task.

There is an extensive test suite using the python `unittest` framework.
It is recommended to use [pytest](https://nose.readthedocs.io/en/latest/) for invoking the test suite.
After having the project set the project in "development mode", at the command line from the top-level directory, issue `pytest`.

Anything having a gradient (all discrete layers and activation functions), as well as some composite networks, has a gradient check run as part of the test suite.
General support for gradient checks is inside [gradient_check.py](pyneural/test/gradient_check.py).
The directory [samples](pyneural/samples) contains several multi-layer networks with code that trains them and shows that the loss decreases during training.


## Using Optimized Numerical Libraries 

When used as a library to implement larger networks, significant variations in execution speed were observed across Ubuntu installations on very similar hardware depending on compiled numerical libraries used.
The performance of numpy, even if it is the same version, varies depending on how numpy and its dependencies were compiled.
Numpy links against a [BLAS](http://www.netlib.org/blas/) implementation.
On macOS that is provided by the Accelerate framework.
On linux very efficient implementations are [OpenBLAS](http://www.openblas.net/) and for Intel CPUs the [Intel OneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html), which was made free on 2017 for certain uses.
On certain Ubuntu installations you will achieve optimum performance when you compile numpy from [source code](https://github.com/numpy/numpy) on your system and configure it to use one of the above libraries before you build it.
You will likely see some performance improvements if you compile OpenBLAS from sources or switch to MKL.

On Ubuntu 16.04 Skylake Core I5 and when using large models and data it was found that using more than one threads with OpenBLAS often *slowed* execution while also consuming more CPU resources.
With MKL that was never observed and there was always some small performance gain with more threads.
For both libraries the shell variable `OMP_NUM_THREADS` controls the number of threads to be used.
If you do not set it explicitly, the default is the number of CPU cores, which can result in very poor CPU utilization for OpenBLAS as explained above.
Always tune and pass `OMP_NUM_THREADS` explicitly.
If you have N jobs to run, it is practically always better to run N concurrent executions with one thread each than one execution with N threads.
Intel MKL appears to be faster than OpenBLAS in the backpropagation path of RNNs.
