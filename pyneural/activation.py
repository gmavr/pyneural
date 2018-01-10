import numpy as np


def sigmoid(x, out=None):
    """ Computes logistic (sigmoid) transfer function
    
    Args:
        x: np.array
        out: if not None, put output in-place
    Returns:
        np.array
    """
    if out is not None:
        # measurably faster in rnn_layer
        np.exp(-x, out=out)
        out += 1.0
        np.divide(1.0, out, out=out)
        return out
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(f, out=None):
    """ Computes the gradient of sigmoid from pre-computed sigmoid function value
    
    Args:
        f: np.array sigmoid function value of original input x.
        out: if not None, put output in-place
    Returns:
        np.array
    """
    if out is not None:
        # measurably faster in rnn_layer
        np.subtract(1.0, f, out=out)
        np.multiply(f, out, out=out)
        return out
    return f * (1.0 - f)


def tanh_grad(f, out=None):
    """ Computes the gradient of tanh from pre-computed tanh value
    
    Args:
        f: np.array tanh function value of original input x.
        out: if not None, put output in-place
    Returns:
        np.array
    """
    if out is not None:
        np.multiply(f, -f, out=out)
        out += 1.0
        return out
    return 1.0 - f*f
    # return 1.0 - np.power(f, 2) # much slower than 1.0 - f*f


def relu(x, out=None):
    """ Computes ReLu
    
    Args:
        x: np.array
        out: if not None, put output in-place
    Returns:
        np.array
    """
    return np.maximum(0.0, x, out=out)  # np.maximum is the fastest of 3 versions
    # return np.clip(x, 0, float('inf'))
    # return np.where(x > 0, x, 0)


def relu_grad(f, out=None):
    """ Computes the gradient of ReLu from pre-computed ReLu function value.
    
    Note: For an element-wise multiplication with relu_grad, it may be more efficient (?) to use indexing:
        g * relu_grad(f) = g(f <= 0)
    Args:
        f: np.array ReLu function value of original input x.
        out: if not None, put output in-place
    """
    if out is not None:
        # which one is faster?
        # np.copyto(out, np.where(f > 0.0, 1.0, 0.0))
        indices = np.greater(f, 0.0)
        out.fill(0.0)
        out[indices] = 1.0
        return out
    return np.where(f > 0.0, 1.0, 0.0)


def relu_clipped(x, out=None):
    return np.clip(x, 0.0, 500.0, out=out)


def relu_clipped_grad(f, out=None):
    if out is not None:
        indices = np.greater(f, 0.0)
        out.fill(0.0)
        out[indices] = 1.0
        return out
    return np.where(f > 0.0, 1.0, 0.0)


def select_activation(activation_name):
    if activation_name == "sigmoid":
        return sigmoid, sigmoid_grad
    if activation_name == "tanh":
        return np.tanh, tanh_grad
    if activation_name == "relu":
        return relu, relu_grad
    if activation_name == "relu_clipped":
        return relu_clipped, relu_clipped_grad
    raise ValueError("illegal activation name")
