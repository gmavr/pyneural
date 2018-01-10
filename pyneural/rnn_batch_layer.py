import numpy as np

import activation as ac
from neural_base import BatchSequencesComponentNN, glorot_init


class RnnBatchLayer(BatchSequencesComponentNN):
    """ Standard Rnn Layer.
    
    First dimension is time, the second dimension is batch (sequence index).
    """
    def __init__(self, dim_d, dim_h, max_seq_length, batch_size, dtype, activation='tanh', bptt_steps=None,
                 grad_clip_thres=None, asserts_on=True):
        self.dim_d, self.dim_h = dim_d, dim_h
        num_params = self.dim_h * self.dim_d + self.dim_h * self.dim_h + self.dim_h
        super(RnnBatchLayer, self).__init__(num_params, max_seq_length, batch_size, dtype)
        # self._curr_batch_seq_dim_length == x.shape[0] of last batch (last x passed in forward_batch(self))
        self._curr_batch_seq_dim_length = 0  # must be 0 before the first iteration
        self.bptt_steps = bptt_steps if bptt_steps is not None else max_seq_length
        assert self.bptt_steps <= self._max_seq_length
        self.w_xh = self.w_hh = self.b = None
        self.dw_xh = self.dw_hh = self.db = None
        self.asserts_on = asserts_on
        if grad_clip_thres:
            self._grad_clip_thres = grad_clip_thres
            assert self._grad_clip_thres > 0.5
        else:
            self._grad_clip_thres = None
        # hs[t, i, :] contains the hidden state for the (t-1)-input element, h[1] is first input hidden state
        # hs[0, i, :] is copied to with self.hs_last[i, :] at the beginning of forward propagation
        self.hs_large = np.empty((self._max_seq_length + 1, self._max_num_sequences, self.dim_h), dtype=dtype)
        self.hs = None
        # hs_last[i, :] contains the last hidden state of the previous mini-batch for the i-th sequence
        self.hs_last = np.zeros((self._max_num_sequences, self.dim_h), dtype=dtype)
        self.data = None
        self.dh_raw = None
        self.dh_raw_large = np.empty((self._max_seq_length, self._max_num_sequences, self.dim_h), dtype=self._dtype)
        self.delta_err_large = np.empty((self._max_seq_length, self._max_num_sequences, self.dim_d), dtype=self._dtype)
        self.dh = np.empty((self._max_num_sequences, self.dim_h), dtype=self._dtype)
        self.dh_next = np.empty((self._max_num_sequences, self.dim_h), dtype=self._dtype)
        self.ac_grad_large = np.empty((self._max_seq_length, self._max_num_sequences, self.dim_h), dtype=self._dtype)
        self.ac_grad = None
        self.hxd_array = np.empty((self.dim_h, self.dim_d), dtype=self._dtype)
        self.hxh_array = np.empty((self.dim_h, self.dim_h), dtype=self._dtype)
        self.activation, self.activation_grad = ac.select_activation(activation)

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({"dim_d": self.dim_d, "dim_h": self.dim_h,
                  "max_seq_length": self._max_seq_length, "max_num_sequences": self._max_num_sequences,
                  "bptt_steps": self.bptt_steps, "activation": self.activation.__name__})
        return d

    def get_dimensions(self):
        return self.dim_d, self.dim_h

    def set_init_h(self, init_h):
        """
        Client must pass here init_h.shape[0] == data.shape[1] in the next invocation of forward_batch().
        The code does not validate this.
        """
        if self.asserts_on:
            assert init_h.ndim == 2
            assert init_h.shape[0] <= self._max_num_sequences
            assert init_h.shape[1] == self.dim_h
            assert init_h.dtype == self._dtype
        self.hs_last[0:init_h.shape[0]] = init_h   # makes a copy, which is desirable

    def reset_last_hidden(self):
        self.hs_last.fill(0.0)

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

    def forward(self, data, seq_lengths):
        if self.asserts_on:
            assert data.ndim == 3
            assert data.shape[0] <= self._max_seq_length
            assert data.shape[1] <= self._max_num_sequences
            assert data.shape[2] == self.dim_d
            assert seq_lengths.shape[0] == data.shape[1]
            # assert seq_lengths.dtype == np.int
            assert np.max(seq_lengths) <= data.shape[0]
            assert 0 <= np.min(seq_lengths)

        self._curr_num_sequences = data.shape[1]
        curr_batch_seq_dim_length = self._curr_batch_seq_dim_length = data.shape[0]
        self.data = data
        self._seq_lengths = seq_lengths

        self._curr_min_seq_length = np.amin(self._seq_lengths)
        self._curr_max_seq_length = np.amax(self._seq_lengths)

        if self.asserts_on:
            self.validate_zero_padding(data)

        # "trim" self.hs_large to proper size (no copy)
        self.hs = self.hs_large[0:(curr_batch_seq_dim_length + 1), 0:self._curr_num_sequences, :]

        # restore the last hidden state of the previous batch (or what was set to via set_init_h())
        self.hs[0] = self.hs_last[0:self._curr_num_sequences]  # makes a copy, which is desirable

        # non-batch was: ((H, D) x (D, T))^T = (T, D) x (D, H) = (T, H)
        # now with batch: (T, B, D) x (D, H) = (T, B, H)
        # broadcasting: (T, B, H) + (H, ) = (T, B, H) + (1, H) -> (T, B, H) + (T, B, H)
        z_partial = np.dot(self.data, self.w_xh.T) + self.b

        # The hidden state passes through the non-linearity, therefore it cannot be optimized
        # as summations over samples. A loop is necessary.
        z_partial_2 = self.dh_next[0:self._curr_num_sequences]  # re-use scratch space and trim
        for t in xrange(self._curr_max_seq_length):
            # non-batch was  z_partial_2:  ((H1, H2) x (H2, )) returns (H1, )
            # now with batch z_partial_2:  ((B, H2) x (H1, H2)^T) returns (B, H1)
            np.dot(self.hs[t], self.w_hh.T, out=z_partial_2)
            z_partial_2 += z_partial[t]
            self.activation(z_partial_2, out=self.hs[t + 1])  # (B, H)

        for s in xrange(self._curr_num_sequences):
            seq_length = seq_lengths[s]
            # remember each sequence's last hidden state
            self.hs_last[s] = self.hs[seq_length, s]
            # The hidden state after each sequence's last was assigned junk values.
            # Zero-out hidden state after each sequence's last.
            # Technically this is not necessary because the upper layer ignores these values anyway, but it may be
            # useful for checking invariants and correctness
            if seq_length < curr_batch_seq_dim_length:
                self.hs[(seq_length+1):(curr_batch_seq_dim_length+1), s] = 0.0

        return self.hs[1:(curr_batch_seq_dim_length + 1)]  # (T, B, H)

    def backwards(self, delta_upper):
        if self.asserts_on:
            assert delta_upper.shape == (self._curr_batch_seq_dim_length, self._curr_num_sequences, self.dim_h)

            # this check is critical for correctness of __back_propagation_loop()
            self.validate_zero_padding(delta_upper)  # DO NOT REMOVE

        curr_max_seq_length = self._curr_max_seq_length

        # (T, B, H)
        if self._curr_num_sequences == self._max_num_sequences:
            # this is the common case (or it should be)
            # "trim" self.dh_raw_large to proper size (no copy) (T, B, H)
            dh_raw = self.dh_raw = self.dh_raw_large[0:self._curr_batch_seq_dim_length]
            ac_grad = self.ac_grad = self.ac_grad_large[0:self._curr_batch_seq_dim_length]
        else:
            # allocate new arrays instead of slicing in 2 dimensions, which could be slower because of elements
            # scattered further away in a larger block of memory
            dh_raw = self.dh_raw = np.empty((self._curr_batch_seq_dim_length, self._curr_num_sequences, self.dim_h),
                                            dtype=self._dtype)
            ac_grad = self.ac_grad = np.empty((self._curr_batch_seq_dim_length, self._curr_num_sequences, self.dim_h),
                                              dtype=self._dtype)
        if curr_max_seq_length < self._curr_batch_seq_dim_length:
            dh_raw[curr_max_seq_length:] = 0.0

        self.activation_grad(self.hs[1:(self._curr_batch_seq_dim_length + 1)], out=ac_grad)

        if curr_max_seq_length <= self.bptt_steps:
            self.__back_propagation_loop(delta_upper, 0, curr_max_seq_length)
        else:
            for start_t in xrange(curr_max_seq_length - self.bptt_steps, -1, -self.bptt_steps):
                self.__back_propagation_loop(delta_upper, start_t, start_t + self.bptt_steps)
            if curr_max_seq_length % self.bptt_steps != 0:
                # first chunk of the sequences in the batch is less than self.bptt_steps samples
                self.__back_propagation_loop(delta_upper, 0, curr_max_seq_length % self.bptt_steps)

        if self.asserts_on:
            self.validate_zero_padding(dh_raw)

        # reduce_sum (T, B, H) to (H, )
        np.sum(dh_raw, axis=(0, 1), out=self.db)
        # we can't easily sum over the T, B dimensions using matrix multiplications, so we use a loop for the first
        # dimension only (time)
        self.dw_xh.fill(0.0)
        self.dw_hh.fill(0.0)
        hxd_array, hxh_array = self.hxd_array, self.hxh_array  # this seems to make a difference in run-time
        for t in xrange(curr_max_seq_length):
            # (H, D) = (H, B) x (B, D) is the sum of outer products (H, 1) x (1, D) over the B sequences at time t
            # self.dw_xh += np.dot(dh_raw[t].T, self.data[t])
            np.dot(dh_raw[t].T, self.data[t], out=hxd_array)
            self.dw_xh += hxd_array
            # (H, H) = (H, B) x (B, H) is the sum of outer products (H, 1) x (1, H) over the B sequences at time t
            # self.dw_hh += np.dot(dh_raw[t].T, self.hs[t])
            np.dot(dh_raw[t].T, self.hs[t], out=hxh_array)
            self.dw_hh += hxh_array

        # non-batch was:  (T, H) x (H, D) = (T, D)
        # now with batch: (T, B, H) x (H, D) = (T, B, D)
        if self._curr_num_sequences == self._max_num_sequences:
            # this is the common case
            # "trim" self.delta_err_large to proper size (no copy)
            delta_err = self.delta_err_large[0:self._curr_batch_seq_dim_length]
            np.dot(dh_raw, self.w_xh, out=delta_err)
        else:
            # note: numpy does not allow as an out= argument a 3-D array "trimmed" in the leading 2 dimensions
            delta_err = np.dot(dh_raw, self.w_xh)

        if self._grad_clip_thres is not None:
            np.clip(self._grad, a_min=-self._grad_clip_thres, a_max=self._grad_clip_thres, out=self._grad)

        return delta_err

    def __back_propagation_loop(self, delta_upper, low_t, high_t):
        """
        Reverse iteration starting from high_t - 1, finishing at low_t, both inclusive.
        Args:
            delta_upper: error signal from upper layer, numpy.array of shape (T, B, output_dimension)
            low_t: low index, inclusive
            high_t: high index, exclusive
        """

        # A thing of beauty: If delta_upper has all 0s after the end of each sequence in the mini-batch (as it should),
        # then the following loop will set self.dh_raw to properly be 0 for these elements.
        # Outside of this method we call self.validate_zero_padding(self.dh_raw) to confirm this.

        if self._curr_num_sequences == self._max_num_sequences:  # slice only if we need to (unclear if matters..)
            dh_next, dh = self.dh_next, self.dh
        else:
            dh_next = self.dh_next[0:self._curr_num_sequences]
            dh = self.dh[0:self._curr_num_sequences]
        ac_grad = self.ac_grad
        dh_next.fill(0.0)
        for t in xrange(high_t - 1, low_t - 1, -1):
            # dh = delta_upper[t] + dh_next  # select at 1st dim from (T, B, H) : (H, ) -> (B, H)
            np.add(delta_upper[t], dh_next, out=dh)
            np.multiply(dh, ac_grad[t], out=self.dh_raw[t])
            # was dh_next = (H1, ) x (H1, H2) = (H2, )
            # dh_next = np.dot(self.w_hh.T, self.dh_raw[t])
            # now (B, H1) x (H1, H2) = (B, H2)
            # dh_next = np.dot(self.dh_raw[t], self.w_hh)
            np.dot(self.dh_raw[t], self.w_hh, out=dh_next)

            # invariant: here we must have dh_raw[i, j, :] == 0 for all i, j with i > _seq_lengths[j]

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


class RnnBatchLayerTime2nd(BatchSequencesComponentNN):
    """ Same as RnnBatchLayer but time indexes the second dimension instead of the first.
    
    RnnBatchLayer is faster. Only use this version if transposing would be necessary and the total cost is more.
    """

    def __init__(self, dim_d, dim_h, max_seq_length, batch_size, dtype, activation='tanh', bptt_steps=None,
                 asserts_on=True):
        self.dim_d = dim_d
        self.dim_h = dim_h
        num_params = self.dim_h * self.dim_d + self.dim_h * self.dim_h + self.dim_h
        super(RnnBatchLayerTime2nd, self).__init__(num_params, max_seq_length, batch_size, dtype)
        self._curr_batch_seq_dim_length = 0  # must be 0 before the first iteration
        self.bptt_steps = bptt_steps if bptt_steps is not None else max_seq_length
        assert self.bptt_steps <= self._max_seq_length
        self.w_xh = self.w_hh = self.b = None
        self.dw_xh = self.dw_hh = self.db = None
        self.asserts_on = asserts_on
        # hs[i, t, :] contains the hidden state for the (t-1)-input element, h[1] is first input hidden state
        # hs[i, 0, :] is copied to with self.hs_last[i, :] at the beginning of forward propagation
        self.hs_large = np.empty((self._max_num_sequences, self._max_seq_length + 1, self.dim_h), dtype=dtype)
        # hs_last[i, :] contains the last hidden state of the previous mini-batch for the i-th sequence
        self.hs = None
        self.hs_last = np.zeros((self._max_num_sequences, self.dim_h), dtype=dtype)
        self.data = None
        self.dh_raw = None
        self.activation, self.activation_grad = ac.select_activation(activation)

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({"dim_d": self.dim_d, "dim_h": self.dim_h,
                  "max_seq_length": self._max_seq_length, "max_num_sequences": self._max_num_sequences,
                  "bptt_steps": self.bptt_steps, "activation": self.activation.__name__})
        return d

    def set_init_h(self, init_h):
        """
        Client must pass here init_h.shape[0] == data.shape[1] in the next invocation of forward_batch().
        The code does not validate this.
        """
        if self.asserts_on:
            assert init_h.ndim == 2
            assert init_h.shape[0] <= self._max_num_sequences
            assert init_h.shape[1] == self.dim_h
            assert init_h.dtype == self._dtype
        self.hs_last[0:init_h.shape[0]] = init_h   # makes a copy, which is desirable

    def reset_last_hidden(self):
        self.hs_last.fill(0.0)

    def forward(self, data, seq_lengths):
        if self.asserts_on:
            assert data.ndim == 3
            assert data.shape[1] <= self._max_seq_length
            assert data.shape[0] <= self._max_num_sequences
            assert data.shape[2] == self.dim_d
            assert seq_lengths.shape[0] == data.shape[0]
            assert seq_lengths.dtype == np.int
            assert np.max(seq_lengths) <= data.shape[1]
            assert 0 <= np.min(seq_lengths)

        self._curr_num_sequences = data.shape[0]
        curr_batch_seq_dim_length = self._curr_batch_seq_dim_length = data.shape[1]
        self.data = data
        self._seq_lengths = seq_lengths

        self._curr_min_seq_length = np.amin(self._seq_lengths)
        self._curr_max_seq_length = np.amax(self._seq_lengths)

        # self.validate_zero_padding(data) does not work for time in 2nd dim

        # "trim" self.hs_large to proper size (no copy)
        self.hs = self.hs_large[0:self._curr_num_sequences, 0:(curr_batch_seq_dim_length + 1), :]

        # restore the last hidden state of the previous batch (or what was set to via set_init_h())
        self.hs[:, 0] = self.hs_last[0:self._curr_num_sequences]  # makes a copy, which is desirable

        # non-batch was: ((H, D) x (D, T))^T returns (T, H)
        # now with batch: (B, T, D) x (D, H) returns (B, T, H)
        # broadcasting: (B, T, H) + (H, ) = (B, T, H) + (1, H) -> (B, T, H) + (B, T, H)
        z_partial = np.dot(self.data, self.w_xh.T) + self.b

        # The hidden state passes through the non-linearity, therefore it cannot be optimized
        # as summations over samples. A loop is necessary.
        for t in xrange(self._curr_max_seq_length):
            # ((H, H) x (B, H)^T) returns (H, B)
            # z_partial_2 = np.dot(self.w_hh, self.hs[:, t].T)
            # ((B, H) x (H, H)^T) returns (B, H)
            z_partial_2 = np.dot(self.hs[:, t], self.w_hh.T)
            self.hs[:, t + 1] = self.activation(z_partial[:, t] + z_partial_2)  # (B, H)

        for s in xrange(self._curr_num_sequences):
            seq_length = seq_lengths[s]
            # remember each sequence's last hidden state
            self.hs_last[s] = self.hs[s, seq_length, :]
            # zero hidden state after each sequence's last
            # was: overwrite / pad hidden state of each sequence with last hidden of the sequence
            if seq_length < curr_batch_seq_dim_length:
                self.hs[s, (seq_length+1):(curr_batch_seq_dim_length+1), :] = 0.0

        return self.hs[:, 1:(curr_batch_seq_dim_length + 1)]  # (B, T, H)

    def backwards(self, delta_upper):
        # use RnnBatchLayer
        raise NotImplementedError

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
