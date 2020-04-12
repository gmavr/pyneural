## Introduction

A very efficient python implementation and framework for training neural networks. It has been explicitly designed with class hierarchy and generic code allowing for modular implementation and wiring of various types of neural network layers. It currently provides standard feed-forward hidden layer, standard and GRU recurrent layers (RNN), bi-directional RNN layer, embedding lookup layer, multi-class logistic (softmax) regression for classification, Conditional Random Fields output layer for sequence modeling classification, squared-error loss output layer for regression. Batching of sequences in parallel has been implemented for the RNN layer, embedding lookup layer and softmax layer. Implementation for four variants of stochastic gradient descend are included, with additional features such as periodic reporting of training loss and periodic evaluation on a held-out data set. It uses [NumPy](http://www.numpy.org/) for efficient matrix operations and matrix views. This framework has been applied successfully to implement state-of-art deep neural networks for Named Entity Recognition (NER) task.


## Design

We consider multi-layer neural networks where the top layer is a scalar loss function. When training such networks for each layer we need expressions for the following two types of derivatives. Given the derivative of the loss function with respect to this layer's output variables (or equivalently the immediately upper layer input variables) we need to compute the:

1. Derivative of the loss function with respect to this layer's trainable parameters
2. Derivative of the loss function with respect to this layer's input variables 

The concatenation of the former across all layers is the derivative of the full model and is used by the stochastic gradient descend algorithm to determine the next value of the full model. The latter is necessary for computing the former: it allows back-propagating the error from the top-most scalar layer incrementally across each layer going to the bottom layer. 

The general framework for supporting the above is in file [neural_layer.py](pyneural/neural_layer.py). Auxiliary classes showing how to combine layers are in [layers.py](pyneural/layers.py).

The framework requires and supports the model and gradient of the full network to be each inside a contiguous memory buffer. The nested components model and gradient are references to the appropriate places inside these memory buffers that are defined at the top-most enclosing network object. These references are set recursively during wiring. During the fitting procedure, the model and gradient buffers are updated strictly in-place and are *never* copied. These architectural features are critical for a very fast training procedure. The fitting procedure itself is inside [sgd.py](pyneural/sgd.py).

This framework favors minimizing execution time almost unconditionally over minimizing memory consumption and implementation complexity. Any intermediate results that can cached to be used in back-propagation are cached instead of recomputed. Vectorization is used heavily. An unusual feature and conscious design choice is that the RNN implementations are *vectorized on the time dimension* as well as the batch dimension. Tensorflow (as of 1.5.0), Cafe2 (as of 0.8.1) do not vectorize in the time dimension. The minimum possible work is done inside the RNN loop, its results are accumulated and the final matrix operations are applied as single operations across all time steps. This makes especially the backward step far more complex to implement but also much faster in a numpy-based implementation.


## Build Instructions

Framework requires python 3.6 or newer. It has been extensively used on macOS (10.13.3 and similar) and ubuntu 16.04. The few python package dependencies are listed in [requirements.txt](requirements.txt). For dependency isolation it is strongly recommended to use a [python3 virtualenv](https://docs.python.org/3.6/library/venv.html) or [conda](https://docs.conda.io/en/latest/). You can build egg or wheel distribution package using the usual `setuptools` mechanisms. For building a wheel package, install `wheel` and then issue `python setup.py bdist_wheel`


## Testing and Correctness

There is an extensive test suite using the python `unittest` framework. It is recommended to use [nose](https://nose.readthedocs.io/en/latest/) for invoking the test suite. From the he top-level directory of the repository it can be invoked from the command line by issuing `python setup.py nosetests` or `python -m unittest discover -s pyneural/test -p "*_test.py"`.

Anything having a gradient (all discrete layers and activation functions), as well as some composite networks, has a gradient check run as part of the test suite. General support for gradient checks is inside [gradient_check.py](pyneural/test/gradient_check.py). The directory [samples](pyneural/samples) contains several multi-layer networks with code that trains them and shows that the loss decreases during training.


## Build Optimizations 

Significant variations in execution speed were observed across  Ubuntu installations on very similar hardware. It appears that the performance of numpy, even if it is the same version, varies depending on how numpy and its dependencies were compiled. Numpy links against a [BLAS](http://www.netlib.org/blas/) implementation. On macOS that is provided by the Accelerate framework. On linux very efficient implementations are [OpenBLAS](http://www.openblas.net/) and for Intel CPUs the [Intel MKL](https://software.intel.com/en-us/mkl), which was made free on 2017 for certain uses. On certain Ubuntu installations you will achieve optimum performance when you compile numpy from [source code](https://github.com/numpy/numpy) on your system and configure it to use one of the above libraries before you build it. You will likely see some performance improvements if you compile OpenBLAS from sources or switch to MKL.

On Ubuntu 16.04 Skylake Core I5 and when using large models and data it was found that using more than one threads with OpenBLAS often *slowed* execution while also consuming more CPU resources. With MKL that was never observed and there was always some small performance gain with more threads. For both libraries the shell variable `OMP_NUM_THREADS` controls the number of threads to be used. If you do not set it explicitly, the default is the number of CPU cores, which can result in very poor CPU utilization for OpenBLAS as explained above. Always tune and pass `OMP_NUM_THREADS` explicitly. If you have N jobs to run, it is practically always better to run N concurrent executions with one thread each than one execution with N threads. Intel MKL appears to be faster than OpenBLAS in the backpropagation path of RNNs.
