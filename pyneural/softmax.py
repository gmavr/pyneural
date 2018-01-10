import numpy as np


def softmax_1d(x):
    """
    Computes the softmax function for one input of type np.array with shape (K, )
    """
    # assert x.ndim == 1

    y = x - np.max(x)
    exps = np.exp(y)
    denom = np.sum(exps)
    return exps / denom


def softmax_1d_opt(x, out=None):
    """
    Same as softmax_1d() but saves memory allocations and allows in-place return value.
    """
    # assert x.ndim == 1

    if out is None:
        y = x - np.max(x)
        np.exp(y, out=y)
        denom = np.sum(y)
        return y / denom

    np.subtract(x, np.max(x), out=out)
    np.exp(out, out=out)
    denom = np.sum(out)
    out /= denom
    return out


def softmax_2d(x):
    """ Computes the softmax function for each row of the input np.array x.

    Args:
        x: np.array of shape (N, K) where each row is one K-dim value that softmax is applied on
    Returns:
        np.array of shape x.shape where each row is the softmax function of corresponding input of x.
    """
    # assert x.ndim == 2

    # the computational cost of the numerical stability trick is about 30% of total cost
    # for softmax on arrays sized 128x10k (128 samples, 10,000 classes) and 128x40k

    # softmax trick for numerical stability:
    # for each sample we subtract the maximum coordinate from all coordinates
    w = np.max(x, axis=1)

    # we want to subtract w[i] from each element in the i-th row of x
    # but because numpy array broadcasting starts from the lower dimension, x and w has incompatible shapes
    # we correct for this by taking the transpose (special case of np.reshape)
    y = (x.T - w).T   # same as y[i, :] = x[i, :] - w[i]

    exps = np.exp(y)
    denom = np.sum(exps, axis=1)
    return (exps.T / denom).T


def softmax_2d_opt(x, out=None):
    """ Computes the softmax function for each row of the input np.array x.

    Same as softmax_2d() but saves 2 memory allocations and allows in-place return value.
    This is measurably faster. For x with x.shape=(128, 200k) the gain is 10% and 20% with using pre-allocated out=.
    """
    # assert x.ndim == 2

    w = np.max(x, axis=1)

    y = (x.T - w).T

    np.exp(y, out=y)
    np.sum(y, axis=1, out=w)

    if out is None:
        return (y.T / w).T

    np.divide(y.T, w, out=out.T)
    return out


def softmax2_2d(x):
    """
    This version of softmax was found to have the same performance for both small (100) and large (70k - 200k) values
    of K as softmax_2d_opt with the matrix transpose operations.
    """
    assert x.ndim == 2

    n = x.shape[0]
    w = np.max(x, axis=1)
    y = x - np.reshape(w, (n, 1))  # (N, K) + (N, 1) -> (N, K) + (N, K)

    exps = np.exp(y)
    denom = np.sum(exps, axis=1)

    return exps / np.reshape(denom, (n, 1))


def softmax_3d(x):
    """ Computes the softmax function on the highest index dimension of the input np.array x.

    Args:
        x: np.array of shape (N, M, K) where the elements in the last dimension is one K-dim value that softmax
        is applied on. Therefore there are N*M such elements
    Returns:
        np.array of shape x.shape where the last dimension holds the softmax function of corresponding input of x.
    """
    # assert x.ndim == 3

    n, m = x.shape[0], x.shape[1]
    w = np.max(x, axis=2)
    y = x - np.reshape(w, (n, m, 1))  # (N, M, K) + (N, M, 1) -> (N, M, K)

    exps = np.exp(y)
    denom = np.sum(exps, axis=2)

    # assert w.shape == (n, m) and denom.shape == (n, m)

    return exps / np.reshape(denom, (n, m, 1))


def softmax_3d_opt(x, out=None):
    """ Computes the softmax function on the highest index dimension of the input np.array x.

    Same as softmax_3d() but saves memory allocations and allows in-place return value.

    Args:
        x: np.array of shape (N, M, K) where the elements in the last dimension is one K-dim value that softmax
        is applied on. Therefore there are N*M such elements
        out: if not None, the output is put in-place to that array
    Returns:
        np.array of shape x.shape where the last dimension holds the softmax function of corresponding input of x.
    """
    # assert x.ndim == 3

    n, m = x.shape[0], x.shape[1]
    w = np.max(x, axis=2)
    y = x - np.reshape(w, (n, m, 1))  # (N, M, K) + (N, M, 1) -> (N, M, K)

    np.exp(y, out=y)
    denom = np.sum(y, axis=2)

    # assert w.shape == (n, m) and denom.shape == (n, m)

    if out is None:
        return y / np.reshape(denom, (n, m, 1))

    np.divide(y, np.reshape(denom, (n, m, 1)), out=out)
    return out


def softmax_2d_no_norm(x):
    """
    Same as softmax_2d() except that it omits the normalization step used for numerical stability
    """
    # assert x.ndim == 2

    exps = np.exp(x)
    denom = np.sum(exps, axis=1)

    return exps / np.reshape(denom, (len(denom), 1))


def test_softmax():
    # softmax(np.array([1, 2])
    expected0 = [0.26894142,  0.73105858]

    # softmax(np.array([1, 2, 3])
    expected1 = [0.09003057, 0.2447285, 0.665241]

    # softmax(np.array([1, 3, 5])
    expected2 = [0.01587624, 0.1173104, 0.8668133]

    rtol = 1e-6
    atol = 1e-10

    dtype = np.float32

    x = np.array([1001.0, 1002.0], dtype=dtype)
    y = softmax_1d(x)
    y1 = softmax_1d_opt(x)
    y2 = np.empty_like(y)
    softmax_1d_opt(x, out=y2)
    assert np.allclose(y, np.array(expected0), rtol=rtol, atol=atol)
    assert np.allclose(y, y1, rtol=rtol, atol=atol)
    assert np.alltrue(np.equal(y1, y2))
    assert y.shape == x.shape

    y = softmax_2d(np.array([[1001.0, 1002.0], [3.0, 4.0]], dtype=dtype))
    assert np.allclose(y, np.array([expected0, expected0], dtype=dtype), rtol=rtol, atol=atol)
    assert y.shape == (2, 2)

    x = np.array([[-1001.0, -1002.0]], dtype=dtype)
    y = softmax_2d(x)
    assert np.allclose(y, np.array([expected0[1], expected0[0]], dtype=dtype), rtol=rtol, atol=atol)
    assert y.shape == (1, 2)

    x = np.array([1, 2, 3], dtype=dtype)
    y = softmax_1d(x)
    assert np.allclose(y, np.array(expected1, dtype=dtype), rtol=rtol, atol=atol)
    assert y.shape == (3, )

    x = np.array([[1, 2, 3]], dtype=dtype)
    y = softmax_2d(x)
    assert np.allclose(y, np.array(expected1, dtype=dtype), rtol=rtol, atol=atol)
    assert y.shape == (1, 3)

    x = np.array([[1, 2, 3], [1, 3, 5]], dtype=dtype)
    y = softmax_2d(x)
    y1 = softmax_2d_opt(x)
    y2 = np.empty_like(y)
    softmax_2d_opt(x, out=y2)
    assert np.allclose(y, np.array([expected1, expected2], dtype=dtype), rtol=rtol, atol=atol)
    assert np.allclose(y, y1, rtol=rtol, atol=atol)
    assert np.alltrue(np.equal(y1, y2))
    assert y.shape == (2, 3)

    x = np.array([[1, 2, 3], [1, 3, 5], [11, 12, 13], [21, 23, 25]], dtype=dtype)
    y = softmax_2d(x)
    y1 = softmax_2d_opt(x)
    y2 = np.empty_like(y)
    softmax_2d_opt(x, out=y2)
    assert np.allclose(y, np.array([expected1, expected2, expected1, expected2], dtype=dtype), rtol=rtol, atol=atol)
    assert np.allclose(y, y1, rtol=rtol, atol=atol)
    assert np.alltrue(np.equal(y1, y2))
    assert y.shape == (4, 3)

    y = softmax_1d(np.array([1, 1, 1, 1]))
    assert np.allclose(y, np.array([0.25, 0.25, 0.25, 0.25], dtype=dtype), rtol=1e-20, atol=1e-20)
    assert y.shape == (4,)

    y = softmax_2d(np.array([[100]], dtype=dtype))
    assert np.allclose(y, np.array([1.0], dtype=dtype), rtol=1e-20, atol=1e-20)
    assert y.shape == (1, 1)

    x = np.empty((3, 2, 3), dtype=dtype)
    x[0] = [[1, 2, 3], [1, 3, 5]]
    x[1] = [[1001, 1003, 1005], [1, 1, 1]]
    x[2] = [[5, 5, 5], [11, 12, 13]]

    expected = np.empty((3, 2, 3))
    expected[0] = np.array([expected1, expected2])
    expected[1] = np.array([expected2, [0.3333333, 0.3333333, 0.3333333]])
    expected[2] = np.array([[0.3333333, 0.3333333, 0.3333333], expected1])

    y = softmax_3d(x)
    y1 = softmax_3d_opt(x)
    y2 = np.empty_like(y)
    softmax_3d_opt(x, out=y2)
    assert np.allclose(y, expected, rtol=rtol, atol=atol)
    assert np.allclose(y, y1, rtol=rtol, atol=atol)
    assert np.alltrue(np.equal(y1, y2))
    assert y.shape == x.shape


if __name__ == "__main__":
    test_softmax()
