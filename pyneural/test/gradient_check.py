import numpy as np

from typing import Callable, Tuple


def gradient_check(f: Callable[[], Tuple[float, np.array]], x: np.array, tolerance: float = 1e-8) -> bool:
    """ Validates the analytically derived gradient by comparing it against the numerical approximation it computes.

    Note: excellent info on gradient checks: http://cs231n.github.io/neural-networks-3/#gradcheck

    Args:
        f: function object that takes no arguments and outputs the scalar loss and its gradient w. r. to x.
            x is the implied argument to that function; a reference to x is contained inside the function object.
        x: point to check the gradient at. It is required to be the SAME reference as inside function object f.
        tolerance: error threshold for failing the gradient check.
            Usually around 1e-8 for np.float64, model and inputs of size up to around 500 and containing values
            with standard deviation up to 0.5.
    Returns:
        bool: check passed or not
    """

    _, grad = f()  # evaluate function once

    # copying the gradient vector is necessary for objects that return a reference to the same gradient vector that they
    # internally retain across invocations
    grad = np.copy(grad)

    h = 1e-5

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # numerical gradient is : grad f(x) = limit_(h->0) (f(x+h)-f(x-h)) / (2h)

        # evaluate f at +h and -h away from point ix on current coordinate
        x[ix] -= h
        fx1, _ = f()

        x[ix] += (2.0 * h)
        fx2, _ = f()

        # restore value
        x[ix] -= h

        # compute the numerical gradient on the current coordinate (partial derivative w.r.to current coordinate)
        numeric_gradient = (fx2 - fx1) / (2.0 * h)

        # Compare gradients
        # somewhat similar to : (|x - y|/(|x| + |y|))
        relative_error = abs(numeric_gradient - grad[ix]) / max(1.0, abs(numeric_gradient), abs(grad[ix]))
        if relative_error > tolerance:
            print("Gradient check failed.")
            print("First gradient error found at coordinate index %s" % str(ix))
            print("Analytical gradient: %.7e\t Numerical gradient: %.7e\t Relative error: %.3e"
                  % (grad[ix], numeric_gradient, relative_error))
            return False

        it.iternext()

    return True


def test_f2():

    class ProjectionFunction(object):

        def __init__(self, x_, dtype_):
            self.x = x_
            self.dtype = dtype_
            self.model = np.random.standard_normal(4).astype(dtype_)

        @staticmethod
        def _pack_parameters(w, b, dst_vec):
            ofs = 0
            length = 3
            dst_vec[ofs:(ofs + length)] = np.reshape(w, (length,))
            ofs += length
            length = 1
            dst_vec[ofs:(ofs + length)] = np.reshape(b, (length,))

        @staticmethod
        def _unpack_parameters(params):
            ofs = 0
            w = np.reshape(params[ofs:(ofs + 3)], (1, 3))
            ofs += 3
            b = np.reshape(params[ofs:(ofs + 1)], (1, ))
            return w, b

        def projection_function(self) -> Tuple[float, np.array]:
            """
            y = sum_x [ W x + b ] with y scalar, W matrix of shape 1xD, b scalar, x of shape D or NxD
            """
            w, b = self._unpack_parameters(self.model)

            if self.x.ndim == 2:
                # if multiple observations, the cost function is their summation
                num_samples = self.x.shape[0]
                dw, db = np.sum(self.x, axis=0), num_samples
                y = np.sum(np.dot(w, self.x.T) + b)
            else:
                dw, db = self.x, 1
                y = np.dot(w, self.x) + b

            grad_packed = np.empty(shape=(4,), dtype=self.dtype)
            self._pack_parameters(dw, db, grad_packed)

            return y, grad_packed

    dtype = np.float64

    x = np.array([2, 4, 7], dtype=dtype)
    pf = ProjectionFunction(x, dtype)
    ret = gradient_check(pf.projection_function, pf.model)
    assert ret

    x = np.array([[-5, 3, 1], [11, -500, 3]], dtype=dtype)
    pf = ProjectionFunction(x, dtype)
    ret = gradient_check(pf.projection_function, pf.model)
    assert ret


if __name__ == "__main__":
    test_f2()
