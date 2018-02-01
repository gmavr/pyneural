## Introduction

A very efficient python implementation and framework for training neural networks. It has been explicitly designed with class hierarchy and generic code allowing for modular implementation and wiring of various types of neural network layers. It currently provides standard feed-forward hidden layer, standard and GRU recurrent layers (RNN), bi-directional RNN layer, embedding lookup layer, multi-class logistic (softmax) regression for classification, Conditional Random Fields output layer for sequence modeling classification, squared-error loss output layer for regression. Batching of sequences in parallel has been implemented for the RNN layer, embedding lookup layer and softmax layer. Implementation for four variants of stochastic gradient descend are included, with additional features such as periodic reporting of training loss and periodic evaluation on a held-out data set. It uses [NumPy](http://www.numpy.org/) for efficient matrix operations and matrix views. This framework has been applied successfully to implement state-of-art deep neural networks for Named Entity Recognition (NER) task.


## Design

We consider multi-layer neural networks where the top layer is a scalar loss function. When training such networks for each layer we need expressions for the following two types of derivatives. Given the derivative of the loss function with respect to this layer's output variables (or equivalently the immediately upper layer input variables) we need to compute the:

1. Derivative of the loss function with respect to this layer's trainable parameters
2. Derivative of the loss function with respect to this layer's input variables 

The concatenation of the former across all layers is the derivative of the full model and is used by the stochastic gradient descend algorithm to determine the next value of the full model. The latter is necessary for computing the former: it allows back-propagating the error from the top-most scalar layer incrementally across each layer going to the bottom layer. 

The general framework for supporting the above is in file `neural_layer.py`. Auxiliary classes showing how to combine layers are in `layers.py`.

The framework requires and supports the model and gradient of the full network to be each inside a contiguous memory buffer. The nested components model and gradient are references to the appropriate places inside these memory buffers that are defined at the top-most enclosing network object. These references are set recursively during wiring. During the fitting procedure, the model and gradient buffers are updated strictly in-place and are *never* copied. These architectural features are critical for a very fast training procedure. The fitting procedure itself is inside `sgd.py`.

An unusual feature of this framework is that the RNN implementations are *vectorized on the time dimension* as well as the batch dimension. Tensorflow (as of 1.5.0), Cafe2 (as of 0.8.1) do not vectorize in the time dimension. The minimum necessary work is kept inside the RNN loop, its results are accumulated and the final matrix operations are applied as single operations across all time steps. This makes especially the backward step far more complex to implement but also much faster in a numpy-based implementation.

## Build Instructions

Works only with python 2.7. Has been extensively used on mac os (10.12.6 and around) and ubuntu 16.04. The few python package dependencies are listed in `requirements.txt`. It is strongly recommended for dependency isolation to use a python [virtualenv](https://virtualenv.pypa.io/en/stable/).

Significant variations in execution speed were observed across  Ubuntu installations on very similar hardware. It appears that performance of numpy, even if it is the same version, varies depending on how numpy and its dependencies was compiled. On certain Ubuntu installations you will achieve optimum performance when you compile numpy from [source code](https://github.com/numpy/numpy) on your system. It appears that numpy 1.12.0 is faster than the later 1.14.0.

## Testing and Correctness

There is an extensive test suite using the python `unittest` framework. The full test suite can be invoked from the command line by issuing `python -m unittest discover -p "*_test.py"` from the top-level directory of the repository.

Anything having a gradient (all discrete layers and activation functions), as well as some composite networks, has a gradient check run as part of the test suite. General support for gradient checks is inside `gradient_check.py`. The directory `samples` contains several multi-layer networks with code that trains them and shows that the loss decreases during training.

