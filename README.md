## Introduction

A very efficient python implementation and framework for training neural networks. It has been explicitly designed with class hierarchy and generic code allowing for modular implementation and wiring of various types of neural network layers. It currently provides standard feed-forward hidden layer, standard and GRU recurrent layers (RNN), bi-directional RNN layer, embedding lookup layer, multi-class logistic (softmax) regression and Conditional Random Fields output layer output layer sfor classification, squared-error loss output layer for regression. Batching of sequences in parallel has been implemented for the RNN layer, embedding lookup layer and softmax layer. Iplementation for four variants of stochastic gradient descend are included, with additional features such as periodic reporting of training loss and periodic evaluation on a held-out data set. It uses [NumPy](http://www.numpy.org/) for efficient matrix operations and matrix views. This framework has been applied successfully to implement state-of-art deep neural networks for Named Entity Recognition (NER) task.


## Design

We consider multi-layer neural networks where the top layer is a scalar loss function. When training such networks for each layer we need expressions for the following two types of derivatives. Given the derivative of the loss function with respect to this layer's output variables (or equivalently the immediately upper layer input variables) we need to compute the:

1. Derivative of the loss function with respect to this layer's trainable parameters
2. Derivative of the loss function with respect to this layer's input variables 

The concatenation of the former across all layers is the derivative of the full model and is used by the stochastic gradient descend algorithm to determine the next value of the full model. The latter is necessary for computing the former: it allows back-propagating the error from the top-most scalar layer incrementally across each layer going to the bottom layer. 

The general framework for supporting the above is in file `neural_layer.py`. Auxiliary classes showing how to combine layers are in `layers.py`.

Note that the framework requires and supports the model and gradient of the full network to be each inside a contiguous memory buffer. The nested components model and gradient are references to the appropriate places inside these memory buffers that are defined at the top-most enclosing network object. These references are set recursively during wiring. During the fitting procedure, the model and gradient buffers are updated strictly in-place and are *never* copied. These architectural features are critical for a very fast training procedure. The fitting procedure itself is inside `sgd.py`.

General support for gradient checks is inside `gradient_check.py`. 


## Build Instructions and Testing

Python 2.7 is necessary. The few additional requirements are listed in `requirements.txt`. It is strongly recommended for dependency isolation to use a python [virtualenv](https://virtualenv.pypa.io/en/stable/). 

There is an extensive test suite using the python `unittest` framework. The full test suite can be invoked from the command line by `python -m unittest discover . "*_test.py"`.


## Correctness

Anything having a gradient (all discrete layers and activation functions), as well as some composite networks, has a gradient check run as part of the test suite. Several multi-layer networks are provided with code that trains them and shows that the loss decreases during training.

