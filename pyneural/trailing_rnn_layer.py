import numpy as np

import activation as ac
from neural_base import ComponentNN
from rnn_layer import RnnLayer
from rnn_batch_layer import RnnBatchLayer, glorot_init
from gru_layer import GruBatchLayer

"""Rnn layers with only the last hidden state is returned.

All implementations present here have class structure shortcomings. The best implementation would be a general
delegation class that encapsulates an rnn layer class (standard or gru), and forwards to it the forward and backwards
methods while adjusting their arguments.
"""


class TrailingRnnLayer(ComponentNN):
    """Standard Rnn Layer where only the last time step hidden state is returned. Typically used in "encoding" schemes.

    Different from regular RnnLayer only in the following:
    1) The hidden state of the last element only is returned (and gradient properly adjusted).
    2) The option to truncate back propagation through time (BPTT) does not exist, so error is propagated to the full
    length of the sequence. (If we did truncated BPTT then the elements beyond the truncation length would contribute
    nothing to the gradient, which in fact might be desirable in certain application).

    Implementing it from scratch instead of in terms of RnnLayer incurs significant code duplication for some small
    gains in readability and run-time execution performance. For dim_h = 100 the execution cost savings is 10%-15%.
    """

    def __init__(self, dim_d, dim_h, max_seq_length, dtype, activation="sigmoid", asserts_on=True):
        self.dim_d, self.dim_h = dim_d, dim_h
        num_params = self.dim_h * self.dim_d + self.dim_h * self.dim_h + self.dim_h
        super(TrailingRnnLayer, self).__init__(num_params, dtype)
        self._seq_length = 0  # must be 0 before the first iteration
        self._max_seq_length = max_seq_length
        self.w_xh = self.w_hh = self.b = None
        self.dw_xh = self.dw_hh = self.db = None
        self.asserts_on = asserts_on
        # hs[t, :] contains the hidden state for the (t-1)-input element
        # hs[0, :] contains the last hidden state of the previous mini-batch
        self.hs = np.empty((self._max_seq_length + 1, self.dim_h), dtype=self._dtype)
        self.reset_last_hidden()  # must initialize initial hidden state to 0 otherwise junk is read at first invocation
        self.data = None
        self.activation, self.activation_grad = ac.select_activation(activation)

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({"dim_d": self.dim_d, "dim_h": self.dim_h, "num_params": self.get_num_p(),
                  "max_seq_length": self._max_seq_length, "activation": self.activation.__name__})
        return d

    def set_init_h(self, init_h):
        if self.asserts_on:
            assert init_h.shape == (self.dim_h, )
            assert init_h.dtype == self._dtype
        self.hs[self._seq_length] = init_h  # this makes a copy of init_h, which is desirable

    def model_normal_init(self, sd):
        assert self._model is not None
        np.copyto(self.w_xh, sd * np.random.standard_normal((self.dim_h, self.dim_d)).astype(self._dtype))
        np.copyto(self.w_hh, sd * np.random.standard_normal((self.dim_h, self.dim_h)).astype(self._dtype))
        self.b.fill(0.0)

    def model_identity_glorot_init(self, scale_factor=0.5):
        assert self._model is not None
        np.copyto(self.w_xh, glorot_init((self.dim_h, self.dim_d)).astype(self._dtype))
        np.multiply(scale_factor, np.eye(self.dim_h, dtype=self._dtype), out=self.w_hh)
        self.b.fill(0.0)

    def reset_last_hidden(self):
        self.hs[self._seq_length] = np.zeros(self.dim_h, dtype=self._dtype)

    def forward(self, x):
        # Implementation-wise the only material difference from regular RnnLayer is that the hidden state at
        # t = seq_length is returned, instead of the full hidden state vector.

        if self.asserts_on:
            assert x.ndim == 2
            assert x.shape[1] == self.dim_d
            assert x.shape[0] <= self._max_seq_length

        # restore the last hidden state of the previous sequence (or what was set to via set_init_h())
        self.hs[0] = self.hs[self._seq_length]  # makes a copy, which is desirable

        self._seq_length = x.shape[0]
        self.data = x

        # ((H, D) x (D, N))^T returns (N, H)
        # broadcasting: (N, H) + (H, ) = (N, H) + (1, H) -> (N, H) + (N, H)
        z_partial = np.dot(self.data, self.w_xh.T) + self.b

        # The hidden state passes through the non-linearity, therefore it cannot be optimized
        # as summations over samples. A loop is necessary.
        z_partial_2 = np.empty(self.dim_h, dtype=self._dtype)
        for t in xrange(self._seq_length):
            # ((H1, H2) x (H2, )) returns (H2, )
            np.dot(self.w_hh, self.hs[t], out=z_partial_2)
            self.activation(z_partial[t] + z_partial_2, out=self.hs[t+1])

        self.y = self.hs[self._seq_length]
        return self.y

    def backwards(self, delta_upper):
        if self.asserts_on:
            assert delta_upper.shape == (self.dim_h, )

        # Implementation-wise the only material difference from regular RnnLayer is that only the delta_upper of the
        # last element in the sequence is back propagated which is equivalent to setting all
        # delta_upper[0:(seq_length-1)] = 0.
        # This implementation just omits the addition with the arrays containing all 0s.

        dh_raw = np.empty((self._seq_length, self.dim_h), dtype=self._dtype)

        t = self._seq_length - 1
        dh = delta_upper
        np.multiply(dh, self.activation_grad(self.hs[t + 1]), out=dh_raw[t])
        dh = np.dot(dh_raw[t], self.w_hh)

        for t in xrange(self._seq_length - 2, -1, -1):
            # delta_upper[t] is my delta_s(t) * W_hy
            # dh_raw[j] is my delta(j, num_steps - 1) defined in formula (13), computed incrementally with (14) :
            np.multiply(dh, self.activation_grad(self.hs[t + 1]), out=dh_raw[t])
            # (H1, ) x (H1, H2) = (H2, )
            np.dot(dh_raw[t], self.w_hh, out=dh)

        # following set arrays in-place (argument out=)
        np.sum(dh_raw, axis=0, out=self.db)
        # (H, N) x (N, D) is the sum of outer products (H, 1) x (1, D) over the N samples
        np.dot(dh_raw.T, self.data, out=self.dw_xh)
        # (H, N) x (N, H) is the sum of outer products (H, 1) x (1, H) over the N samples
        np.dot(dh_raw.T, self.hs[0:self._seq_length], out=self.dw_hh)

        # (N, H) x (H, D) == (N, D)
        return np.dot(dh_raw, self.w_xh)

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


class TrailingRnnLayer2(RnnLayer):
    """ Identical functionality as TrailingRnnLayer but implemented in terms of RnnLayer.

    Performance-wise, TrailingRnnLayer is better.
    """
    def __init__(self, dim_d, dim_h, max_seq_length, dtype, activation, bptt_steps=None):
        super(TrailingRnnLayer2, self).__init__(dim_d, dim_h, max_seq_length, dtype, activation, bptt_steps)
        self.y = None

    def forward(self, x):
        y1 = super(TrailingRnnLayer2, self).forward(x)
        # XXX this sets the parent class self.y to something else, RnnLayer is resistant to that.
        # This is implementation inheritance at its ugliest. However, it is too much here to replace inheritance with
        # delegation and forward everything
        assert self._seq_length > 0
        self.y = y1[self._seq_length-1]
        return self.y

    def backwards(self, delta_upper):
        assert delta_upper.shape == (self.dim_h,)
        delta_upper1 = np.zeros((self._seq_length, self.dim_h), self._dtype)
        delta_upper1[self._seq_length-1] = delta_upper
        return super(TrailingRnnLayer2, self).backwards(delta_upper1)


class TrailingRnnBatchLayer(RnnBatchLayer):
    """
    First dimension is time, the second dimension is batch (sequence index).

    Uses implementation inheritance to avoid boilerplate from composition and forwarding.
    """
    def __init__(self, dim_d, dim_h, max_seq_length, max_batch_size, dtype, activation="tanh", asserts_on=True):
        super(TrailingRnnBatchLayer, self).__init__(dim_d, dim_h, max_seq_length, max_batch_size,
                                                    dtype=dtype, activation=activation, bptt_steps=max_seq_length,
                                                    asserts_on=asserts_on)
        self.y = None

    def get_max_seq_length_out(self):
        return 1

    def get_seq_lengths_out(self):
        return np.ones(self._curr_num_sequences, dtype=np.int32)

    def forward(self, data, seq_lengths):
        y1 = super(TrailingRnnBatchLayer, self).forward(data, seq_lengths)
        self.y = np.empty((self._curr_num_sequences, self.dim_h), dtype=self._dtype)
        for i in xrange(self._curr_num_sequences):
            if seq_lengths[i] == 0:
                # This case is not well-defined because the assumption is that the layer always returns 1 hidden state.
                # Return all 0s predictions and treat the upper layer error vector as 0 (even if it is not).
                # This is similar to drop-out applied.
                self.y[i].fill(0.0)
            else:
                self.y[i] = y1[seq_lengths[i] - 1, i, :]
        return self.y

    def backwards(self, delta_upper):
        if self.asserts_on:
            assert delta_upper.shape == (self._curr_num_sequences, self.dim_h)
        delta_upper1 = np.zeros((self._curr_seq_length_dim_max, self._curr_num_sequences, self.dim_h), self._dtype)
        for i in xrange(self._curr_num_sequences):
            # if self._seq_lengths[i] == 0 leave delta_upper1 all 0s (even if it was not supplied as all 0s,
            # which is legitimate to happen). This is similar to drop-out applied. See also forward_batch().
            if self._seq_lengths[i] > 0:
                delta_upper1[self._seq_lengths[i] - 1, i, :] = delta_upper[i]
        return super(TrailingRnnBatchLayer, self).backwards(delta_upper1)


class TrailingGruBatchLayer(GruBatchLayer):
    """
    First dimension is time, the second dimension is batch (sequence index).

    Uses implementation inheritance to avoid boilerplate from composition and forwarding.

    This is essentially an identical wrapper as TrailingRnnBatchLayer.
    """

    def __init__(self, dim_d, dim_h, max_seq_length, max_batch_size, dtype, asserts_on=True):
        super(TrailingGruBatchLayer, self).__init__(dim_d, dim_h, max_seq_length, max_batch_size, dtype, asserts_on)
        self.y = None

    def get_max_seq_length_out(self):
        return 1

    def get_seq_lengths_out(self):
        return np.ones(self._curr_num_sequences, dtype=np.int32)

    def forward(self, data, seq_lengths):
        y1 = super(TrailingGruBatchLayer, self).forward(data, seq_lengths)
        self.y = np.empty((self._curr_num_sequences, self.dim_h), dtype=self._dtype)
        for i in xrange(self._curr_num_sequences):
            if seq_lengths[i] == 0:
                # This case is not well-defined because the assumption is that the layer always returns 1 hidden state.
                # Return all 0s predictions and treat the upper layer error vector as 0 (even if it is not).
                # This is similar to drop-out applied.
                self.y[i].fill(0.0)
            else:
                self.y[i] = y1[seq_lengths[i] - 1, i, :]
        return self.y

    def backwards(self, delta_upper):
        if self.asserts_on:
            assert delta_upper.shape == (self._curr_num_sequences, self.dim_h)
        delta_upper1 = np.zeros((self._curr_seq_length_dim_max, self._curr_num_sequences, self.dim_h), self._dtype)
        for i in xrange(self._curr_num_sequences):
            # if self._seq_lengths[i] == 0 leave delta_upper1 all 0s (even if it was not supplied as all 0s,
            # which is legitimate to happen). This is similar to drop-out applied. See also forward_batch().
            if self._seq_lengths[i] > 0:
                delta_upper1[self._seq_lengths[i] - 1, i, :] = delta_upper[i]
        return super(TrailingGruBatchLayer, self).backwards(delta_upper1)
