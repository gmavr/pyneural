import numpy as np

import pyneural.activation as ac
from pyneural.neural_base import ComponentNN, glorot_init


class NeuralLayer(ComponentNN):
    """ Standard fully connected neural layer with connections with optional activation (transfer) function.

    It is allowed to set activation=None, in which case a projection layer is produced.
    """
    def __init__(self, dim_x, dim_y, dtype, activation="tanh", asserts_on=True):
        self.dim_x = dim_x
        self.dim_y = dim_y
        super(NeuralLayer, self).__init__((self.dim_x + 1) * self.dim_y, dtype)
        self.w = self.b = None
        self.dw = self.db = None
        self.asserts_on = asserts_on
        self.x = None
        self.delta_err = None
        if activation:
            self.activation, self.activation_grad = ac.select_activation(activation)
        else:
            self.activation, self.activation_grad = None, None

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({"dim_x": self.dim_x, "dim_y": self.dim_y,
                  "activation": self.activation.__name__ if self.activation is not None else 'None'})
        return d

    def model_normal_init(self, sd):
        assert self._model is not None
        np.copyto(self.w, sd * np.random.standard_normal((self.dim_y, self.dim_x)).astype(self._dtype))
        self.b.fill(0.0)

    def model_glorot_init(self):
        assert self._model is not None
        np.copyto(self.w, glorot_init((self.dim_y, self.dim_x)).astype(self._dtype))
        self.b.fill(0.0)

    def forward_single(self, x):
        assert x.shape == (self.dim_x, )

        self.x = x
        z = np.dot(self.w, self.x) + self.b
        self.y = self.activation(z)

        return self.y

    def backwards_single(self, delta_err):
        assert delta_err.shape == (self.dim_y, )
        assert self.y.shape == (self.dim_y, )

        delta_err2 = delta_err * self.activation_grad(self.y)  # element-wise product
        np.outer(delta_err2, self.x, out=self.dw)
        np.copyto(self.db, delta_err2)

        self.delta_err = np.dot(self.w.T, delta_err2)  # (Dx, Dy) x (Dy, )
        return self.dw, self.db, self.delta_err

    def forward(self, x):
        if self.asserts_on:
            assert x.ndim == 2
            assert x.shape[1] == self.dim_x
            assert x.dtype == self._dtype

        self.x = x

        # ((Dy, Dx) x (Dx, N))^T == (N, Dx) x (Dx, Dy) returns (N, Dy)
        # broadcasting: (N, Dy) + (Dy, ) = (N, Dy) + (1, Dy) -> (N, Dy) + (N, Dy)
        z = np.dot(self.x, self.w.T) + self.b
        if self.activation:
            self.y = self.activation(z)
        else:
            self.y = z

        return self.y

    def backwards(self, delta_err_in):
        if self.asserts_on:
            assert delta_err_in.shape == (self.x.shape[0], self.dim_y)
            assert self.y.shape == (self.x.shape[0], self.dim_y)
            assert delta_err_in.dtype == self._dtype

        if self.activation:
            # saving 1 additional memory allocation is almost certainly not worth 3 python function calls:
            # delta_err2 = np.empty((self.x.shape[0], self.dim_y), dtype=self._dtype)
            # self.activation_grad(self.y, out=delta_err2)
            # np.multiply(delta_err_in, delta_err2, out=delta_err2)
            delta_err2 = delta_err_in * self.activation_grad(self.y)  # element-wise product
        else:
            delta_err2 = delta_err_in
        # (Dy, N) x (N, Dx) is the sum of outer products (Dy, 1) x (1, Dx) over the N samples
        np.dot(delta_err2.T, self.x, out=self.dw)
        np.sum(delta_err2, axis=0, out=self.db)  # (Dy, )

        self.delta_err = np.dot(delta_err2, self.w)  # (N, Dy) x (Dy, Dx)
        return self.delta_err

    def __unpack_model_or_grad(self, params):
        ofs = 0
        w = np.reshape(params[ofs:(ofs + self.dim_x * self.dim_y)], (self.dim_y, self.dim_x))
        ofs += self.dim_x * self.dim_y
        b = np.reshape(params[ofs:(ofs + self.dim_y)], (self.dim_y, ))

        if self.asserts_on:
            w.view().shape = (self.dim_x * self.dim_y,)
            b.view().shape = (self.dim_y, )
            assert np.shares_memory(w, params)

        return w, b

    def _set_model_references_in_place(self):
        self.w, self.b = self.__unpack_model_or_grad(self._model)

    def _set_gradient_references_in_place(self):
        self.dw, self.db = self.__unpack_model_or_grad(self._grad)

    def get_built_model(self):
        return np.concatenate((self.w.flatten(), self.b))

    def get_built_gradient(self):
        return np.concatenate((self.dw.flatten(), self.db))

    @staticmethod
    def get_number_parameters_static(dim_x, dim_y):
        return (dim_x + 1) * dim_y
