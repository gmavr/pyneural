import numpy as np

from typing import Type, Union
import pyneural.activation as ac
from pyneural.neural_base import ComponentNN, glorot_init


class RnnLayer(ComponentNN):

    def __init__(self, dim_d: int, dim_h: int, max_seq_length: int, dtype: Union[Type[np.float32], Type[np.float64]],
                 activation="tanh", bptt_steps=None, grad_clip_thres=None, asserts_on=True) -> None:
        """ Standard Recurrent Neural Layer

        Args:
            dim_d: input dimension
            dim_h: hidden state dimension
            max_seq_length: maximum sequence length for any sequence that will be presented in forward and backwards.
            dtype: numpy type of all parameters and inputs: np.float32 or np.float64
            activation: activation (transfer) function string name
            bptt_steps: length of truncated backpropagation. If not None, at the back-propagation loop, the error for
                any sample is back-propagated at most bptt_steps steps instead of at most sequence length steps. This
                does not save any computation, but can limit the damage from exploding gradients, if they occur.
            grad_clip_thres: maximum allowed value for any coordinate of the gradient. If not None, then any gradient
                component is clipped at that value when it exceeds that value
            asserts_on: perform invariant and consistency assertions. Recommended to set to False only at final steps of
                training large models
        """
        self.dim_d, self.dim_h = dim_d, dim_h
        num_p = self.dim_h * self.dim_d + self.dim_h * self.dim_h + self.dim_h
        super().__init__(num_p, dtype)
        self.bptt_steps = bptt_steps if bptt_steps is not None else max_seq_length
        self._seq_length = 0  # must be 0 before the first iteration
        self._max_seq_length = max_seq_length
        assert self.bptt_steps <= self._max_seq_length
        self.w_xh = self.w_hh = self.b = None
        self.dw_xh = self.dw_hh = self.db = None
        self.asserts_on = asserts_on
        if grad_clip_thres:
            self._grad_clip_thres = grad_clip_thres
            assert self._grad_clip_thres > 0.5
        else:
            self._grad_clip_thres = None
        # hs[t, :] contains the hidden state for the (t-1)-input element, h[1] is first input hidden state
        # hs[0, :] contains the last hidden state of the previous sequence
        self.hs = np.empty((self._max_seq_length + 1, self.dim_h), dtype=dtype)
        self.reset_last_hidden()  # must initialize initial hidden state to 0 otherwise junk is read at first invocation
        self.x = None
        self.delta_err_large = np.empty((self._max_seq_length, self.dim_d), dtype=self._dtype)
        self.dh_raw_large = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.dh_raw = None
        self.dh = np.empty(self.dim_h, dtype=self._dtype)  # allocate once instead of per backprop
        self.dim_h_buf = np.empty(self.dim_h, dtype=self._dtype)  # (H, )
        self.ac_grad_large = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.ac_grad = None
        self.activation, self.activation_grad = ac.select_activation(activation)

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({"dim_d": self.dim_d, "dim_h": self.dim_h,
                  "max_seq_length": self._max_seq_length, "grad_clip_thres": self._grad_clip_thres,
                  "bptt_steps": self.bptt_steps, "activation": self.activation.__name__})
        return d

    def get_max_seq_length(self) -> int:
        return self._max_seq_length

    def get_last_seq_length(self) -> int:
        return self._seq_length

    def set_init_h(self, init_h: np.array) -> None:
        if self.asserts_on:
            assert init_h.shape == (self.dim_h, )
            assert init_h.dtype == self._dtype
        self.hs[self._seq_length] = init_h  # this makes a copy of init_h, which is desirable

    def reset_last_hidden(self) -> None:
        self.hs[self._seq_length].fill(0.0)

    def model_normal_init(self, sd) -> None:
        assert self._model is not None
        np.copyto(self.w_xh, sd * np.random.standard_normal((self.dim_h, self.dim_d)).astype(self._dtype))
        np.copyto(self.w_hh, sd * np.random.standard_normal((self.dim_h, self.dim_h)).astype(self._dtype))
        self.b.fill(0.0)

    def model_glorot_init(self) -> None:
        assert self._model is not None
        np.copyto(self.w_xh, glorot_init((self.dim_h, self.dim_d)).astype(self._dtype))
        np.copyto(self.w_hh, glorot_init((self.dim_h, self.dim_h)).astype(self._dtype))
        self.b.fill(0.0)

    def model_identity_glorot_init(self, scale_factor=0.5) -> None:
        assert self._model is not None
        np.copyto(self.w_xh, glorot_init((self.dim_h, self.dim_d)).astype(self._dtype))
        np.multiply(scale_factor, np.eye(self.dim_h, dtype=self._dtype), out=self.w_hh)
        self.b.fill(0.0)

    def forward(self, x):
        if self.asserts_on:
            assert x.ndim == 2
            assert x.shape[1] == self.dim_d
            assert x.shape[0] <= self._max_seq_length

        # restore the last hidden state of the previous sequence (or what was set to via set_init_h())
        self.hs[0] = self.hs[self._seq_length]  # makes a copy, which is desirable

        self._seq_length = x.shape[0]
        self.x = x

        # ((H, D) x (D, N))^T = (N, D) x (D, H) = (N, H)
        # broadcasting: (N, H) + (H, ) = (N, H) + (1, H) -> (N, H) + (N, H)
        z_partial = np.dot(self.x, self.w_xh.T) + self.b

        # The hidden state passes through the non-linearity, therefore it cannot be optimized
        # as summations over samples. A loop is necessary.
        z_partial_2 = self.dim_h_buf
        for t in range(self._seq_length):
            # ((H1, H2) x (H2, )) returns (H1, )
            np.dot(self.w_hh, self.hs[t], out=z_partial_2)
            z_partial_2 += z_partial[t]
            self.activation(z_partial_2, out=self.hs[t + 1])

        self.y = self.hs[1:(self._seq_length+1)]
        return self.y

    def backwards(self, delta_upper: np.array) -> np.array:
        if self.asserts_on:
            assert delta_upper.shape == (self._seq_length, self.dim_h)

        # "trim" self.dh_raw_large to proper size (no copy)
        dh_raw = self.dh_raw = self.dh_raw_large[0:self._seq_length]  # (N, H)
        self.ac_grad = self.ac_grad_large[0:self._seq_length]  # (N, H)

        self.activation_grad(self.hs[1:(self._seq_length + 1)], out=self.ac_grad)

        if self._seq_length <= self.bptt_steps:
            self.__back_propagation_loop(delta_upper, 0, self._seq_length)
        else:
            for start_t in range(self._seq_length - self.bptt_steps, -1, -self.bptt_steps):
                self.__back_propagation_loop(delta_upper, start_t, start_t + self.bptt_steps)
            if self._seq_length % self.bptt_steps != 0:
                # first chunk in the batch has fewer than self.bptt_steps samples
                self.__back_propagation_loop(delta_upper, 0, self._seq_length % self.bptt_steps)

        # following set arrays in-place (argument out=)
        np.sum(dh_raw, axis=0, out=self.db)
        # (H, N) x (N, D) is the sum of outer products (H, 1) x (1, D) over the N samples
        np.dot(dh_raw.T, self.x, out=self.dw_xh)
        # (H, N) x (N, H) is the sum of outer products (H, 1) x (1, H) over the N samples
        np.dot(dh_raw.T, self.hs[0:self._seq_length], out=self.dw_hh)

        # (N, H) x (H, D) = (N, D)
        delta_err = self.delta_err_large[0:self._seq_length]  # "trim" self.delta_err_large to proper size (no copy)
        np.dot(dh_raw, self.w_xh, out=delta_err)

        if self._grad_clip_thres is not None:
            np.clip(self._grad, a_min=-self._grad_clip_thres, a_max=self._grad_clip_thres, out=self._grad)

        return delta_err

    def __back_propagation_loop(self, delta_upper: np.array, low_t: int, high_t: int):
        """
        Reverse iteration starting from high_t - 1, finishing at low_t, both inclusive.
        Populates self.dh_raw[low_t:high_t]
        Args:
            delta_upper: error signal from upper layer
            low_t: low index, inclusive
            high_t: high index, exclusive
        """
        dh_next = self.dim_h_buf
        dh_next.fill(0.0)
        dh = self.dh
        ac_grad = self.ac_grad
        for t in range(high_t - 1, low_t - 1, -1):
            # delta_upper[t] is my delta_s(t) * W_hy
            # dh_raw[j] is my delta(j, num_steps - 1) defined in formula (13), computed incrementally with (14) :
            np.add(delta_upper[t], dh_next, out=dh)
            np.multiply(dh, ac_grad[t], out=self.dh_raw[t])  # element-wise multiplication (H, )
            # (H1, ) x (H1, H2) = (H2, )
            np.dot(self.dh_raw[t], self.w_hh, out=dh_next)  # (H, )

    def __unpack_model_or_grad(self, params):
        hxd = self.dim_h * self.dim_d
        hxh = self.dim_h * self.dim_h
        ofs = 0
        w_xh = np.reshape(params[ofs:(ofs + hxd)], (self.dim_h, self.dim_d))
        ofs += hxd
        w_hh = np.reshape(params[ofs:(ofs + hxh)], (self.dim_h, self.dim_h))
        ofs += hxh
        b = np.reshape(params[ofs:(ofs + self.dim_h)], (self.dim_h, ))
        ofs += self.dim_h

        if self.asserts_on:
            # verify no memory copy
            # http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
            w_xh.view().shape = (hxd, )
            w_hh.view().shape = (hxh, )
            b.view().shape = (self.dim_h, )
            assert np.shares_memory(w_xh, params)

        return w_xh, w_hh, b

    def _set_model_references_in_place(self):
        self.w_xh, self.w_hh, self.b = self.__unpack_model_or_grad(self._model)

    def _set_gradient_references_in_place(self):
        self.dw_xh, self.dw_hh, self.db = self.__unpack_model_or_grad(self._grad)

    def get_built_model(self):
        return np.concatenate((self.w_xh.flatten(), self.w_hh.flatten(), self.b))

    def get_built_gradient(self):
        return np.concatenate((self.dw_xh.flatten(), self.dw_hh.flatten(), self.db))


class RnnLayer2(ComponentNN):
    """ Standard Recurrent Neural Layer

    Minor implementation difference from RnnLayer. Recommended to use RnnLayer, not this one.

    Marginally slower than RnnLayer because of one additional element-wise multiplication, but more conformant to my
    derivation document.
    (6.61e-4 vs 6.65e-4 per batch in fit_run.run_rnn_sgd_long for dim_d, dim_h, dim_k = (100, 400, 6400))
    """
    def __init__(self, dim_d, dim_h, max_seq_length, dtype, activation="tanh", bptt_steps=None, grad_clip_thres=None,
                 asserts_on=True):
        self.dim_d, self.dim_h = dim_d, dim_h
        num_p = self.dim_h * self.dim_d + self.dim_h * self.dim_h + self.dim_h
        super().__init__(num_p, dtype)
        self.bptt_steps = bptt_steps if bptt_steps is not None else max_seq_length
        self._seq_length = 0  # must be 0 before the first iteration
        self._max_seq_length = max_seq_length
        assert self.bptt_steps <= self._max_seq_length
        self.w_xh = self.w_hh = self.b = None
        self.dw_xh = self.dw_hh = self.db = None
        self.asserts_on = asserts_on
        assert not grad_clip_thres
        # hs[t, :] contains the hidden state for the (t-1)-input element, h[1] is first input hidden state
        # hs[0, :] contains the last hidden state of the previous sequence
        self.hs = np.empty((self._max_seq_length + 1, self.dim_h), dtype=dtype)
        self.reset_last_hidden()  # must initialize initial hidden state to 0 otherwise junk is read at first invocation
        self.data = None
        self.delta_err_large = np.empty((self._max_seq_length, self.dim_d), dtype=self._dtype)
        self.dh_raw_large = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.dh_raw = None
        self.dh = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.dh_next = np.empty(self.dim_h, dtype=self._dtype)  # (H, )
        self.dim_h_buf = np.empty(self.dim_h, dtype=self._dtype)  # (H, )
        self.ac_grad_large = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.ac_grad = None
        self.activation, self.activation_grad = ac.select_activation(activation)

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({"dim_d": self.dim_d, "dim_h": self.dim_h,
                  "max_seq_length": self._max_seq_length,
                  "bptt_steps": self.bptt_steps, "activation": self.activation.__name__})
        return d

    def get_max_seq_length(self):
        return self._max_seq_length

    def get_last_seq_length(self):
        return self._seq_length

    def set_init_h(self, init_h):
        if self.asserts_on:
            assert init_h.shape == (self.dim_h,)
            assert init_h.dtype == self._dtype
        self.hs[self._seq_length] = init_h  # this makes a copy of init_h, which is desirable

    def reset_last_hidden(self):
        self.hs[self._seq_length].fill(0.0)

    def model_normal_init(self, sd):
        assert self._model is not None
        np.copyto(self.w_xh, sd * np.random.standard_normal((self.dim_h, self.dim_d)).astype(self._dtype))
        np.copyto(self.w_hh, sd * np.random.standard_normal((self.dim_h, self.dim_h)).astype(self._dtype))
        self.b.fill(0.0)

    def model_glorot_init(self):
        assert self._model is not None
        np.copyto(self.w_xh, glorot_init((self.dim_h, self.dim_d)).astype(self._dtype))
        np.copyto(self.w_hh, glorot_init((self.dim_h, self.dim_h)).astype(self._dtype))
        self.b.fill(0.0)

    def model_identity_glorot_init(self, scale_factor=0.5):
        assert self._model is not None
        np.copyto(self.w_xh, glorot_init((self.dim_h, self.dim_d)).astype(self._dtype))
        np.multiply(scale_factor, np.eye(self.dim_h, dtype=self._dtype), out=self.w_hh)
        self.b.fill(0.0)

    def forward(self, x):
        if self.asserts_on:
            assert x.ndim == 2
            assert x.shape[1] == self.dim_d
            assert x.shape[0] <= self._max_seq_length

        # restore the last hidden state of the previous sequence (or what was set to via set_init_h())
        self.hs[0] = self.hs[self._seq_length]  # makes a copy, which is desirable

        self._seq_length = x.shape[0]
        self.data = x

        # ((H, D) x (D, N))^T = (N, D) x (D, H) = (N, H)
        # broadcasting: (N, H) + (H, ) = (N, H) + (1, H) -> (N, H) + (N, H)
        z_partial = np.dot(self.data, self.w_xh.T) + self.b

        # The hidden state passes through the non-linearity, therefore it cannot be optimized
        # as summations over samples. A loop is necessary.
        z_partial_2 = self.dim_h_buf
        for t in range(self._seq_length):
            # ((H1, H2) x (H2, )) returns (H1, )
            np.dot(self.w_hh, self.hs[t], out=z_partial_2)
            z_partial_2 += z_partial[t]
            self.activation(z_partial_2, out=self.hs[t + 1])

        self.y = self.hs[1:(self._seq_length + 1)]
        return self.y

    def backwards(self, delta_upper):
        if self.asserts_on:
            assert delta_upper.shape == (self._seq_length, self.dim_h)

        seq_length = self._seq_length

        # "trim" self.dh_raw_large and self.ac_grad to proper size (no copy)
        dh_raw = self.dh_raw = self.dh_raw_large[0:seq_length]  # (N, H)
        self.ac_grad = self.ac_grad_large[0:seq_length]  # (N, H)

        self.activation_grad(self.hs[1:(seq_length + 1)], out=self.ac_grad)

        if self._seq_length <= self.bptt_steps:
            self.__back_propagation_loop(delta_upper, 0, self._seq_length)
        else:
            for start_t in range(self._seq_length - self.bptt_steps, -1, -self.bptt_steps):
                self.__back_propagation_loop(delta_upper, start_t, start_t + self.bptt_steps)
            if self._seq_length % self.bptt_steps != 0:
                # first chunk in the batch has fewer than self.bptt_steps samples
                self.__back_propagation_loop(delta_upper, 0, self._seq_length % self.bptt_steps)

        # dh_raw is my uppercase Delta matrix multiplied by diag(ac_grad)
        np.multiply(self.dh[0:seq_length], self.ac_grad, out=dh_raw)  # (N, H)

        np.sum(dh_raw, axis=0, out=self.db)
        # (H, N) x (N, D) is the sum of outer products (H, 1) x (1, D) over the N samples
        np.dot(dh_raw.T, self.data, out=self.dw_xh)
        # (H, N) x (N, H) is the sum of outer products (H, 1) x (1, H) over the N samples
        np.dot(dh_raw.T, self.hs[0:seq_length], out=self.dw_hh)

        # (N, H) x (H, D) = (N, D)
        # delta_err = np.dot(dh_raw, self.w_xh)
        delta_err = self.delta_err_large[0:seq_length]  # "trim" self.delta_err_large to proper size (no copy)
        np.dot(dh_raw, self.w_xh, out=delta_err)

        return delta_err

    def __back_propagation_loop(self, delta_upper, low_t, high_t):
        """
        Reverse iteration starting from high_t - 1, finishing at low_t, both inclusive.
        Populates self.dh[low_t:high_t]
        Args:
            delta_upper: error signal from upper layer
            low_t: low index, inclusive
            high_t: high index, exclusive
        """
        # dh_next[t] is my delta(t,T) x DH_{t} computed recursively by (13)
        # dh[t] is my delta(t, T)
        dh_next = self.dh_next
        self.dh[high_t - 1] = delta_upper[high_t - 1]
        for t in range(high_t - 2, low_t - 1, -1):
            np.multiply(self.dh[t + 1], self.ac_grad[t+1], out=self.dim_h_buf)  # (H, )
            # (H1, ) x (H1, H2) = (H2, )
            np.dot(self.dim_h_buf, self.w_hh, out=dh_next)
            np.add(delta_upper[t], dh_next, out=self.dh[t])

    def __unpack_model_or_grad(self, params):
        hxd = self.dim_h * self.dim_d
        hxh = self.dim_h * self.dim_h
        ofs = 0
        w_xh = np.reshape(params[ofs:(ofs + hxd)], (self.dim_h, self.dim_d))
        ofs += hxd
        w_hh = np.reshape(params[ofs:(ofs + hxh)], (self.dim_h, self.dim_h))
        ofs += hxh
        b = np.reshape(params[ofs:(ofs + self.dim_h)], (self.dim_h, ))
        ofs += self.dim_h

        if self.asserts_on:
            # verify no memory copy
            # http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
            w_xh.view().shape = (hxd, )
            w_hh.view().shape = (hxh, )
            b.view().shape = (self.dim_h, )
            assert np.shares_memory(w_xh, params)

        return w_xh, w_hh, b

    def _set_model_references_in_place(self):
        self.w_xh, self.w_hh, self.b = self.__unpack_model_or_grad(self._model)

    def _set_gradient_references_in_place(self):
        self.dw_xh, self.dw_hh, self.db = self.__unpack_model_or_grad(self._grad)

    def get_built_model(self):
        return np.concatenate((self.w_xh.flatten(), self.w_hh.flatten(), self.b))

    def get_built_gradient(self):
        return np.concatenate((self.dw_xh.flatten(), self.dw_hh.flatten(), self.db))
