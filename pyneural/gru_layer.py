import numpy as np

import pyneural.activation as ac
import pyneural.misc_cy as misc_cy
from pyneural.neural_base import ComponentNN, BatchSequencesComponentNN, glorot_init, validate_x_and_lengths


class GruLayer(ComponentNN):
    """Gated Recurrent Unit.

    New state is computed here as: h_t = (1 - z_t) * h_{t-1} + z_t * candidate_t
    while in tensorflow as: h_t = z_t * h_{t-1} + (1 - z_t) * candidate_t
    We can exchange models between these implementations by negating all parameters of the update gate.
    """

    def __init__(self, dim_d, dim_h, max_seq_length, dtype, asserts_on=True):
        self.dim_d, self.dim_h = dim_d, dim_h
        num_p = 3 * self.dim_h * self.dim_d + 3 * self.dim_h * self.dim_h + 3 * self.dim_h
        super(GruLayer, self).__init__(num_p, dtype)
        self.asserts_on = asserts_on
        self._seq_length = 0  # must be 0 before the first iteration
        self._max_seq_length = max_seq_length
        self.w = self.u = self.b = None
        self.dw = self.du = self.db = None
        # inputs to activation functions, dimensionality: (L, 3*H)
        self.sigmoid_in = np.empty((self._max_seq_length, 2 * self.dim_h), dtype=self._dtype)
        self.tanh_in = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        # outputs from activation functions, dimensionality: (L, 3*H)
        self.sigmoid_out = np.empty((self._max_seq_length, 2 * self.dim_h), dtype=self._dtype)
        self.hs_tilde = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        # self.dhs_tilde_dh = np.empty((self._max_seq_length - 1, self.dim_h, self.dim_h), dtype=self._dtype)
        # self.wb_partial = np.empty((self._max_seq_length, 3 * self.dim_h), dtype=self._dtype)
        self.u_partial = np.empty(3 * self.dim_h, dtype=self._dtype)
        self.r_prod_h = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        # hs[t, :] contains the hidden state for the (t-1)-input element, h[1] is first input hidden state
        # hs[0, :] contains the last hidden state of the previous sequence
        self.hs = np.empty((self._max_seq_length + 1, self.dim_h), dtype=dtype)
        self.sigmoid_grad = np.empty((self._max_seq_length, 2 * self.dim_h), dtype=self._dtype)
        self.tanh_grad = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.dh = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.dh2 = np.empty((self._max_seq_length, 3 * dim_h), dtype=self._dtype)
        self.dh_next = np.empty(self.dim_h, dtype=self._dtype)  # (H, )
        self.one_minus_z = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.h_tilde_minus_h = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.z_prod_grad = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.h_minus_h_prod_grad = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.h_prod_grad = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        self.rhr = np.empty((self._max_seq_length - 1, dim_h, dim_h), dtype=self._dtype)
        self.delta_err_large = np.empty((self._max_seq_length, self.dim_d), dtype=self._dtype)
        self.buf_h = np.empty(dim_h, dtype=self._dtype)
        self.buf_hh = np.empty((dim_h, dim_h), dtype=self._dtype)
        self.reset_last_hidden()  # must initialize initial hidden state to 0 otherwise junk is read at first invocation
        self.x = None

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({"dim_d": self.dim_d, "dim_h": self.dim_h, "max_seq_length": self._max_seq_length})
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
        np.copyto(self.w, sd * np.random.standard_normal((3 * self.dim_h, self.dim_d)).astype(self._dtype))
        np.copyto(self.u, sd * np.random.standard_normal((3 * self.dim_h, self.dim_h)).astype(self._dtype))
        # initialize to -1.0, 1.0 to (mostly) not update and not reset
        self.b[:self.dim_h] = -1.0
        self.b[self.dim_h:(2 * self.dim_h)] = 1.0
        self.b[(2 * self.dim_h):] = 0.0

    def model_glorot_init(self):
        assert self._model is not None
        for w in (self.w[:self.dim_h], self.w[self.dim_h:(2*self.dim_h)], self.w[(2*self.dim_h):]):
            np.copyto(w, glorot_init((self.dim_h, self.dim_d)).astype(self._dtype))
        for u in (self.u[:self.dim_h], self.u[self.dim_h:(2*self.dim_h)], self.u[(2*self.dim_h):]):
            np.copyto(u, glorot_init((self.dim_h, self.dim_h)).astype(self._dtype))
        # initialize to -1.0, 1.0 to (mostly) not update and not reset
        self.b[:self.dim_h] = -1.0
        self.b[self.dim_h:(2 * self.dim_h)] = 1.0
        self.b[(2 * self.dim_h):] = 0.0

    def forward_batch_debug(self, x):
        """ More readable, less optimized version than forward() which should be faster.

        It turns out that its running time is similar to or faster than the more "optimized" version.
        The "optimized" version avoids memory allocations at the expense of more python function invocations.
        """
        if self.asserts_on:
            assert x.ndim == 2
            assert x.shape[1] == self.dim_d
            assert x.shape[0] <= self._max_seq_length

        # restore the last hidden state of the previous sequence (or what was set to via set_init_h())
        self.hs[0] = self.hs[self._seq_length]  # makes a copy, which is desirable

        self._seq_length = x.shape[0]
        self.x = x

        sigmoid_in = self.sigmoid_in[0:self._seq_length]
        tanh_in = self.tanh_in[0:self._seq_length]
        sigmoid_out = self.sigmoid_out[0:self._seq_length]
        dim_h = self.dim_h

        u_zr, u_h = self.u[0:(2 * dim_h)], self.u[(2 * dim_h):]
        wb = np.dot(self.x, self.w.T) + self.b
        wb_zr = wb[:, 0:(2 * dim_h)]
        wb_h = wb[:, (2 * dim_h):]

        for t in range(self._seq_length):
            # ((2 * H, H) x (H, )) returns (2 * H, )
            sigmoid_in[t] = wb_zr[t] + np.dot(u_zr, self.hs[t])
            sigmoid_out[t] = ac.sigmoid(sigmoid_in[t])
            z_t = sigmoid_out[t, 0:dim_h]
            r_t = sigmoid_out[t, (1*dim_h):]
            self.one_minus_z[t] = np.subtract(1.0, z_t)
            self.r_prod_h[t] = r_t * self.hs[t]
            tanh_in[t] = wb_h[t] + np.dot(u_h, self.r_prod_h[t])
            self.hs_tilde[t] = np.tanh(tanh_in[t])
            self.hs[t + 1] = self.one_minus_z[t] * self.hs[t] + z_t * self.hs_tilde[t]

        self.y = self.hs[1:(self._seq_length+1)]
        return self.y

    def forward(self, x):
        if self.asserts_on:
            assert x.ndim == 2
            assert x.shape[1] == self.dim_d
            assert x.shape[0] <= self._max_seq_length

        # restore the last hidden state of the previous sequence (or what was set to via set_init_h())
        self.hs[0] = self.hs[self._seq_length]  # makes a copy, which is desirable

        self._seq_length = x.shape[0]
        self.x = x

        sigmoid_in = self.sigmoid_in[0:self._seq_length]
        tanh_in = self.tanh_in[0:self._seq_length]
        sigmoid_out = self.sigmoid_out[0:self._seq_length]
        dim_h = self.dim_h

        # pre-allocating wb_partial once and re-using it in each forward() invocation makes no measurable difference in
        # running time (not surprising given how many further operations are performed in this method)
        # ((3*H, D) x (D, N))^T = (N, D) x (D, 3*H) returns (N, 3*H)
        # broadcasting: (N, 3*H) + (3*H, ) = (N, 3*H) + (1, 3*H) -> (N, 3*H) + (N, 3*H)
        wb_partial = np.dot(self.x, self.w.T) + self.b

        # The hidden state passes through the non-linearity, therefore it cannot be optimized
        # as summations over samples. A loop is necessary.
        u_partial = self.u_partial
        buf_h = self.buf_h  # marginally faster to re-use buffer instead of allocating
        for t in range(self._seq_length):
            # ((2 * H, H) x (H, )) returns (2 * H, )
            np.dot(self.u[0:(2*dim_h)], self.hs[t], out=u_partial[0:(2*dim_h)])
            np.add(wb_partial[t, 0:(2*dim_h)], u_partial[0:(2*dim_h)], out=sigmoid_in[t])
            ac.sigmoid(sigmoid_in[t], out=sigmoid_out[t])
            # sigmoid_out[t, 0:dim_h].fill(1.0)  # for disabling update gate z_t
            z_t = sigmoid_out[t, 0:dim_h]  # select (1, H) from (T, 3*H)
            # sigmoid_out[t, (1*dim_h):].fill(1.0)  # for disabling reset gate r_t
            r_t = sigmoid_out[t, (1*dim_h):]
            np.subtract(1.0, z_t, out=self.one_minus_z[t])
            r_prod_h = self.r_prod_h[t]
            np.multiply(r_t, self.hs[t], out=r_prod_h)
            # ((H, H) x (H, )) returns (H, )
            np.dot(self.u[(2*dim_h):], r_prod_h, out=u_partial[(2 * dim_h):])
            np.add(wb_partial[t, (2*dim_h):], u_partial[(2*dim_h):], out=tanh_in[t])
            np.tanh(tanh_in[t], out=self.hs_tilde[t])
            np.multiply(z_t, self.hs_tilde[t], out=self.hs[t + 1])
            np.multiply(self.hs[t], self.one_minus_z[t], out=buf_h)
            self.hs[t + 1] += buf_h

        self.y = self.hs[1:(self._seq_length+1)]
        return self.y

    def backwards(self, delta_upper):
        if self.asserts_on:
            assert delta_upper.shape == (self._seq_length, self.dim_h)

        seq_length = self._seq_length
        dim_h = self.dim_h

        # "trim" to proper sizes (no copies)
        sigmoid_out = self.sigmoid_out[0:seq_length]  # (T, 2*H)
        sigmoid_grad = self.sigmoid_grad[0:seq_length]
        hs_tilde = self.hs_tilde[0:seq_length]
        tanh_grad = self.tanh_grad[0:seq_length]

        ac.sigmoid_grad(sigmoid_out, out=sigmoid_grad)  # (T, 2 * H)
        ac.tanh_grad(hs_tilde, out=tanh_grad)  # (T, H)

        h_tilde_minus_h = self.h_tilde_minus_h[0:seq_length]
        np.subtract(hs_tilde, self.hs[0:seq_length], out=h_tilde_minus_h)

        z_prod_grad = self.z_prod_grad[0:seq_length]  # (T, H)
        np.multiply(tanh_grad, sigmoid_out[:, 0:dim_h], out=z_prod_grad)

        h_minus_h_prod_grad = self.h_minus_h_prod_grad[0:seq_length]  # (T, H)
        np.multiply(sigmoid_grad[:, 0:dim_h], h_tilde_minus_h, out=h_minus_h_prod_grad)

        h_prod_grad = self.h_prod_grad[0:seq_length]  # (T, H)
        np.multiply(sigmoid_grad[:, dim_h:(2 * dim_h)], self.hs[0:seq_length], out=h_prod_grad)

        if seq_length > 0:
            rhr = self.rhr[0:(seq_length-1)]
            self.vect_element_wise_matrix_plus_diag(self.h_prod_grad[1:seq_length], self.u[dim_h:(2 * dim_h)],
                                                    sigmoid_out[1:seq_length, (1 * dim_h):(2 * dim_h)], rhr)
            self.__back_propagation_loop(delta_upper, 0, seq_length)

        u_h = self.u[(2*dim_h):, ]

        # computing c2, c3 as "vectorized" (T, H, H) arrays outside of the loop is faster for dim_h == 100, but is
        # slower for dim_h = 200, 400 than computing on (H, H) inside the loop..

        # c3 = self.buf_t_h2[0:seq_length]  # (T, H, H)
        #
        # # compute: diag(z_prod_grad) x U_h = c3
        # # from (T, H) to (T, H, 1)
        # c1 = np.reshape(z_prod_grad, (seq_length, dim_h, 1))
        # # broadcasting: (T, H, 1) hadamard (H, H) -> (T, H, H) hadamard (T, H, H)
        # np.multiply(c1, u_h, out=c3)
        #
        # # compute c3 x diag(h_prod_grad)
        # # from (T, H) to (T, 1, H)
        # c2 = np.reshape(h_prod_grad, (seq_length, 1, dim_h))
        # # broadcasting: (T, H, H) hadamard (T, 1, H) -> (T, H, H) hadamard (T, H, H)
        # np.multiply(c3, c2, out=c3)  # (T, H, H)

        # fill dh2
        dh2 = self.dh2[0:seq_length]
        np.multiply(self.dh[0:seq_length], z_prod_grad, out=dh2[:, (2*dim_h):])  # (T, H)
        np.multiply(self.dh[0:seq_length], h_minus_h_prod_grad, out=dh2[:, :dim_h])  # (T, H)
        # (t, H1), (t, H1, H2) -> (t, H2)
        for t in range(seq_length):
            # (H1, 1) hadamard (H1, H2) vs ((H1, ) hadamard (H2, H1)).T
            # version with reshaping is marginally slower than with transpose
            # np.multiply(np.reshape(z_prod_grad[t], (dim_h, 1)), u_h, out=self.buf_hh)
            np.multiply(z_prod_grad[t], u_h.T, out=self.buf_hh.T)
            # (H2) hadamard (H1, H2) -> (H1, H2) hadamard (H1, H2) row vector repeated
            np.multiply(self.buf_hh, h_prod_grad[t], out=self.buf_hh)
            np.dot(self.dh[t], self.buf_hh, out=dh2[t, dim_h:(2 * dim_h)])
            # np.dot(self.dh[t], c3[t], out=dh2[t, dim_h:(2*dim_h)])  # (H, ) x (H, H) = (1, H)

        np.sum(dh2, axis=0, out=self.db)

        np.dot(dh2.T, self.x, out=self.dw)  # (3*H, T) x (T, D)

        np.dot(dh2[:, 0:(2*dim_h)].T, self.hs[0:seq_length], out=self.du[:(2*dim_h)])  # (2*H, T) x (T, H)
        r_dot_h = self.r_prod_h[0:seq_length]
        np.dot(dh2[:, (2*dim_h):].T, r_dot_h, out=self.du[(2*dim_h):])  # (H, T) x (T, H)

        # (T, 3 * H) x (3 * H, D) = (T, D)
        delta_err = self.delta_err_large[0:seq_length]
        np.dot(dh2, self.w, out=delta_err)

        return delta_err

    def vect_element_wise_matrix(self, n_vec, mat, out):
        """
        For each t of N items: Construct matrix M of shape (H1, H2) by repeating vector n_vec[t] H2 times as H2 column
        vectors. Compute element-wise product of M and mat (both are of shape (H1, H2), H1=H2). Add the n_rvec[t]
        vector to the diagonal of the result.

        Args:
            n_vec: N vectors of size D (to be repeated as column vectors Z times)
            mat: matrix of shape (D, Z)
            out: array of shape (N, D, Z) to hold the results
        """
        if self.asserts_on:
            assert n_vec.ndim == 2 and mat.ndim == 2
            assert n_vec.shape[1] == mat.shape[0]
        n, dim = n_vec.shape
        # from (N, D) to (N, D, 1), broadcasting will "repeat column vector" Z times
        n_vec = np.reshape(n_vec, (n, dim, 1))
        # broadcasting: (N, D, 1) hadamard (D, Z) -> (N, D, Z) hadamard (N, D, Z)
        np.multiply(n_vec, mat, out=out)

    def vect_element_wise_matrix_plus_diag(self, n_vec, mat, n_rvec, out):
        """
        For each of N items: Repeat column vector n_vec[t] H2 times, compute element-wise product of vec and mat
        (both have shape (H1, H2), H=H1=H2), add n_rvec[t] vector to its diagonal

        Args:
            n_vec: N vectors of size H1 (to be repeated as column vectors H2 times)
            mat: matrix of shape (H1, H2)
            n_rvec: N vectors of size H to be added to the diagonal
            out: array of shape (N, H1, H2) to hold the results
        """
        if self.asserts_on:
            assert n_vec.ndim == 2
            n, h = n_vec.shape
            assert mat.shape == (h, h)
            assert n_rvec.shape == n_vec.shape
            assert out.shape == (n, h, h)
        n, h = n_vec.shape
        # (N, H1, 1) hadamard (H1, H2) -> (N, H1, H2) hadamard (N, H1, H2).
        c = n_vec.reshape((n, h, 1))
        np.multiply(c, mat, out=out)
        # Following can't be vectorized. Equivalent Cython implementation is slower.
        for i in range(n):
            out[i].flat[0::(h + 1)] += n_rvec[i]  # the most efficient way to add to a matrix's diagonal

    def __back_propagation_loop(self, delta_upper, low_t, high_t):
        """
        Reverse iteration starting from high_t - 1, finishing at low_t, both inclusive.
        Populates self.dh[low_t:high_t]
        Args:
            delta_upper: error signal from upper layer, shape (L, H)
            low_t: low index, inclusive
            high_t: high index, exclusive
        """
        # dh_next[t] is my delta(t,T) x DH_{t} computed recursively by (13)
        # dh[t] is my delta(t, T)
        # dht[t] is my DH_{t}
        dim_h = self.dim_h
        buf_h = self.buf_h
        buf_h_2 = np.empty(dim_h, dtype=self._dtype)
        buf_hh = self.buf_hh
        u_z = self.u[0:dim_h]
        u_h = self.u[(2 * dim_h):]
        dh_next = self.dh_next
        self.dh[high_t - 1] = delta_upper[high_t - 1]
        # Moving multiplication with self.rhr here would produce (H, H1) x (T-1, H1, H2) = (H, T-1, H2), but we'd like
        # (T-1, H, H2) for fast access.
        for t in range(high_t - 2, low_t - 1, -1):
            dh = self.dh[t + 1]
            np.multiply(dh, self.h_minus_h_prod_grad[t + 1], out=buf_h)
            np.dot(buf_h, u_z, out=dh_next)
            np.multiply(dh, self.one_minus_z[t+1], out=buf_h)
            dh_next += buf_h
            np.multiply(dh, self.z_prod_grad[t + 1], out=buf_h)
            # Following has no dependence on a value that is changed within the loop and therefore can be moved as one
            # operation outside the loop. It turns out that doing so destroys performance because either a transpose or
            # selection of non-contiguous elements must occur in a much larger matrix.
            np.dot(u_h, self.rhr[t], out=buf_hh)
            np.dot(buf_h, buf_hh, out=buf_h_2)
            dh_next += buf_h_2
            np.add(delta_upper[t], dh_next, out=self.dh[t])

    # def __compute_dht(self):
    #     seq_length = self._seq_length
    #     dim_h = self.dim_h
    #
    #     rhr = self.rhr[0:(seq_length-1)]
    #     # compute (h_t-1) hadamard sigmoid_grad U_r + diag_r
    #     # T-1 times: diag(H) x (H, H) = (T-1, H, H)
    #     self.vect_element_wise_matrix(self.h_dot_grad[1:seq_length], self.u[dim_h:(2*dim_h)], rhr)
    #     rhr += self.diag_r[1:seq_length]
    #
    #     # compute z hadamard tanh_grad U_h
    #     zrhr = np.empty((seq_length - 1, dim_h, dim_h), dtype=self._dtype)
    #     # T-1 times: diag(H) x (H, H) = (T-1, H, H)
    #     self.vect_element_wise_matrix(self.z_dot_grad[1:seq_length], self.u[(2 * dim_h):], zrhr)
    #
    #     z_dhs_tilde_dh = self.dhs_tilde_dh[0:(seq_length - 1)]
    #
    #     for t in range(seq_length - 1):
    #         # for each t: from (T-1, H, H) select (H, H), then (H, H) x (H, H)
    #         np.dot(zrhr[t], rhr[t], out=z_dhs_tilde_dh[t])
    #
    #     # compute dht
    #     dht = self.dht[0:(seq_length-1)]  # (T-1, H, H)
    #
    #     # T-1 times: diag(H) x (H, H) = (T-1, H, H)
    #     self.vect_element_wise_matrix(self.h_minus_h_dot_grad[1:seq_length], self.u[0:dim_h], dht)
    #
    #     dht += self.diag_1_minus_z[1:seq_length]
    #     dht += z_dhs_tilde_dh

    def __unpack_model_or_grad(self, params):
        hxh = self.dim_h * self.dim_h
        hxd = self.dim_h * self.dim_d
        of1 = 0
        of2 = 3 * hxd
        w = np.reshape(params[of1:of2], (3 * self.dim_h, self.dim_d))
        of1 = of2
        of2 += 3 * hxh
        u = np.reshape(params[of1:of2], (3 * self.dim_h, self.dim_h))
        of1 = of2
        of2 += 3 * self.dim_h
        b = np.reshape(params[of1:of2], (3 * self.dim_h, ))

        if self.asserts_on:
            # verify no memory copy
            # http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
            w.view().shape = (3 * hxd, )
            u.view().shape = (3 * hxh, )
            b.view().shape = (3 * self.dim_h, )
            assert np.shares_memory(w, params)

        return w, u, b

    def _set_model_references_in_place(self):
        self.w, self.u, self.b = self.__unpack_model_or_grad(self._model)

    def _set_gradient_references_in_place(self):
        self.dw, self.du, self.db = self.__unpack_model_or_grad(self._grad)

    def get_built_model(self):
        return np.concatenate((self.w.flatten(), self.u.flatten(), self.b))

    def get_built_gradient(self):
        return np.concatenate((self.dw.flatten(), self.du.flatten(), self.db))


class GruBatchLayer(BatchSequencesComponentNN):
    """ Gated Recurrent Unit. Batch of sequences.

    Only forward propagation implemented. It is substantially faster per sequence than the non-batched version, even for
    small batch sizes.

    First dimension is time, the second dimension is batch (sequence index).
    """

    def __init__(self, dim_d, dim_h, max_seq_length, max_batch_size, dtype, asserts_on=True):
        self.dim_d, self.dim_h = dim_d, dim_h
        num_p = 3 * self.dim_h * self.dim_d + 3 * self.dim_h * self.dim_h + 3 * self.dim_h
        super(GruBatchLayer, self).__init__(num_p, max_seq_length, max_batch_size, dtype)
        self.asserts_on = asserts_on
        self._seq_length = 0  # must be 0 before the first iteration
        self._max_seq_length = max_seq_length
        self.w = self.u = self.b = None
        self.dw = self.du = self.db = None
        # inputs to activation functions, dimensionality: (L, B, 3*H)
        self.sigmoid_in = np.empty((max_seq_length, self._max_num_sequences, 2 * dim_h), dtype=dtype)
        self.tanh_in = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        # outputs from activation functions, dimensionality: (L, B, 3*H)
        self.sigmoid_z_out = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        self.sigmoid_r_out = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        self.tanh_out = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        self.u_zr_partial = np.empty((self._max_num_sequences, 2 * dim_h), dtype=dtype)
        self.u_h_partial = np.empty((self._max_num_sequences, dim_h), dtype=dtype)
        self.r_prod_h = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        # hs[t, i, :] contains the hidden state for the (t-1)-input element, h[1] is first input hidden state
        # hs[0, i, :] is copied to with self.hs_last[i, :] at the beginning of forward propagation
        self.hs_large = np.empty((max_seq_length + 1, self._max_num_sequences, dim_h), dtype=dtype)
        self.hs = None
        # hs_last[i, :] contains the last hidden state of the previous mini-batch for the i-th sequence
        self.hs_last = np.zeros((self._max_num_sequences, dim_h), dtype=dtype)
        self.one_minus_z = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        self.buf_b_h = np.empty((self._max_num_sequences, dim_h), dtype=dtype)
        self.sigmoid_z_grad = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        self.sigmoid_r_grad = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        self.tanh_grad = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        self.dh = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        self.dh2 = np.empty((max_seq_length, self._max_num_sequences, 3 * dim_h), dtype=dtype)
        self.dh_next = np.empty((self._max_num_sequences, dim_h), dtype=dtype)  # (H, )
        self.h_tilde_minus_h = np.empty((max_seq_length, self._max_num_sequences,  dim_h), dtype=dtype)
        self.z_prod_grad = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        self.h_minus_h_prod_grad = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        self.h_prod_grad = np.empty((max_seq_length, self._max_num_sequences, dim_h), dtype=dtype)
        self.reset_last_hidden()  # must initialize initial hidden state to 0 otherwise junk is read at first invocation
        self.x = None

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({"dim_d": self.dim_d, "dim_h": self.dim_h, "max_seq_length": self._max_seq_length,
                  "max_num_sequences": self._max_num_sequences})
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

    def forward(self, x, seq_lengths):
        if self.asserts_on:
            validate_x_and_lengths(x, self._max_seq_length, self._max_num_sequences, seq_lengths)
            assert x.ndim == 3
            assert x.shape[2] == self.dim_d

        self._set_lengths(x, seq_lengths)

        curr_num_sequences = self._curr_num_sequences
        curr_seq_length_dim_max = self._curr_seq_length_dim_max
        curr_max_seq_length = self._curr_max_seq_length

        if self.asserts_on:
            self._validate_zero_padding(x)

        # "trim" to proper sizes (no copies)
        self.x = x[0:self._curr_max_seq_length]  # optimization
        self.hs = self.hs_large[0:(curr_seq_length_dim_max + 1), 0:curr_num_sequences]  # correctness

        # restore the last hidden state of the previous batch (or what was set to via set_init_h())
        self.hs[0] = self.hs_last[0:curr_num_sequences]  # makes a copy, which is desirable

        # non-batch was: (T, D) x (D, 3*H) = (T, 3*H)
        # now with batch: (T, B, D) x (D, 3*H) = (T, B, 3*H)
        # broadcasting: (T, B, 3*H) + (3*H, ) = (T, B, 3*H) + (1, 1, 3*H) -> (T, B, 3*H)
        wb_partial = np.dot(self.x, self.w.T) + self.b

        # "trim" to proper sizes (no copies)
        sigmoid_in = self.sigmoid_in[0:curr_max_seq_length, 0:curr_num_sequences]
        sigmoid_z_out = self.sigmoid_z_out[0:curr_max_seq_length, 0:curr_num_sequences]  # (T, B, H)
        sigmoid_r_out = self.sigmoid_r_out[0:curr_max_seq_length, 0:curr_num_sequences]
        tanh_in = self.tanh_in[0:curr_max_seq_length, 0:curr_num_sequences]
        tanh_out = self.tanh_out[0:curr_max_seq_length, 0:curr_num_sequences]
        u_zr_partial = self.u_zr_partial[0:curr_num_sequences]  # (B, 2*H)
        u_h_partial = self.u_h_partial[0:curr_num_sequences]  # (B, H)
        one_minus_z = self.one_minus_z[0:curr_max_seq_length, 0:curr_num_sequences]
        buf_b_h = self.buf_b_h[0:curr_num_sequences]
        dim_h = self.dim_h

        # The hidden state passes through the non-linearity, therefore it cannot be optimized
        # as summations over samples. A loop is necessary.
        for t in range(self._curr_max_seq_length):
            # non-batch was  u_zr_partial: ((2*H, H) x (H, )) == (2*H, )
            # now with batch u_zr_partial: ((B, H) x (2*H, H)^T) == (B, 2*H)
            np.dot(self.hs[t], self.u[0:(2*dim_h)].T, out=u_zr_partial)  # (B, 2*H)
            np.add(wb_partial[t, :, 0:(2*dim_h)], u_zr_partial, out=sigmoid_in[t])
            # select (1, B, H) from (T, B, 2*H) collapses to (B, H)
            ac.sigmoid(sigmoid_in[t, :, 0:dim_h], out=sigmoid_z_out[t])  # (B, H)
            ac.sigmoid(sigmoid_in[t, :, (1*dim_h):(2*dim_h)], out=sigmoid_r_out[t])  # (B, H)
            z_t = sigmoid_z_out[t]
            r_t = sigmoid_r_out[t]
            np.subtract(1.0, z_t, out=one_minus_z[t])
            r_prod_h = self.r_prod_h[t, 0:curr_num_sequences]  # (B, H)
            np.multiply(r_t, self.hs[t], out=r_prod_h)  # (B, H)
            # non-batch was  u_h_partial: (H1, H2) x (H2, ) == (H1, )
            # now with batch u_h_partial: ((B, H2) x (H1, H2)^T) == (B, H1)
            np.dot(r_prod_h, self.u[(2*dim_h):].T, out=u_h_partial)  # (B, H)
            np.add(wb_partial[t, :, (2*dim_h):], u_h_partial, out=tanh_in[t])
            np.tanh(tanh_in[t], out=tanh_out[t])  # (1, B, H)
            np.multiply(z_t, tanh_out[t], out=self.hs[t + 1])
            np.multiply(self.hs[t], one_minus_z[t], out=buf_b_h)
            self.hs[t + 1] += buf_b_h

        for s in range(curr_num_sequences):
            seq_length = seq_lengths[s]
            # remember each sequence's last hidden state
            self.hs_last[s] = self.hs[seq_length, s]
            # The hidden state after each sequence's last was assigned junk values.
            # Zero-out hidden state after each sequence's last. Required by base class contract.
            if seq_length < curr_seq_length_dim_max:
                self.hs[(seq_length+1):(curr_seq_length_dim_max+1), s] = 0.0

        return self.hs[1:(curr_seq_length_dim_max + 1)]  # (T, B, H)

    def vect_element_wise_matrix_plus_diag(self, n_vec, mat, n_rvec, out):
        """
        See non-batch version for this extension to batched.
        For each of N items: Apply broadcasting to create implicit (N. B, H, H) and compute element-wise product of
        vec and mat.
        Add n_rvec[t] vector to its diagonal.
        Note: For identifying each of the two dimensions of size H, we use notation H1, H2.

        Args:
            n_vec: matrix of shape (N, B, H1)
                N, B vectors of size H1 (to be repeated as column vectors H2 times)
            mat: matrix of shape (H1, H2)
            n_rvec: N, B vectors of size H to be added to the diagonal of (H1, H2).
            out: array of shape (N, B, H1, H2) to hold the results
        """
        if self.asserts_on:
            assert n_vec.ndim == 3
            n, b, h = n_vec.shape
            assert mat.shape == (h, h)
            assert n_rvec.shape == n_vec.shape
            assert out.shape == (n, b, h, h)
        # (N, B, H1, 1) hadamard (H1, H2) -> (N, B, H1, H2) hadamard (N, B, H1, H2).
        c = np.expand_dims(n_vec, axis=3)
        np.multiply(c, mat, out=out)
        misc_cy.add_to_diag_batch(out, n_rvec)
        # Following can't be vectorized. Replaced with equivalent but much faster cython implementation above.
        # n, b, h = n_vec.shape
        # for i in range(n):
        #     for j in range(b):
        #         out[i, j].flat[0::(h + 1)] += n_rvec[i, j]  # the most efficient way to add to a matrix's diagonal

    def backwards(self, delta_upper):
        if self.asserts_on:
            assert delta_upper.shape == (self._curr_seq_length_dim_max, self._curr_num_sequences, self.dim_h)

        # This check is too critical to be omitted. It is expensive to compute but not very much compared to the cost of
        # the rest of this implementation. It does not verify correctness of this layer but instead that a contract-
        # compliant error signal is fed from the higher layer.
        self._validate_zero_padding(delta_upper)  # DO NOT REMOVE! If not 0-padded, we return wrong error signal

        curr_max_seq_length = self._curr_max_seq_length
        curr_num_sequences = self._curr_num_sequences
        dim_h = self.dim_h

        # "trim" to proper sizes (no copies)
        sigmoid_z_out = self.sigmoid_z_out[0:curr_max_seq_length, 0:curr_num_sequences]  # (T, B, H)
        sigmoid_r_out = self.sigmoid_r_out[0:curr_max_seq_length, 0:curr_num_sequences]
        sigmoid_z_grad = self.sigmoid_z_grad[0:curr_max_seq_length, 0:curr_num_sequences]
        sigmoid_r_grad = self.sigmoid_r_grad[0:curr_max_seq_length, 0:curr_num_sequences]
        tanh_out = self.tanh_out[0:curr_max_seq_length, 0:curr_num_sequences]
        tanh_grad = self.tanh_grad[0:curr_max_seq_length, 0:curr_num_sequences]
        hs = self.hs[0:curr_max_seq_length, 0:curr_num_sequences]

        ac.sigmoid_grad(sigmoid_z_out, out=sigmoid_z_grad)
        ac.sigmoid_grad(sigmoid_r_out, out=sigmoid_r_grad)
        ac.tanh_grad(tanh_out, out=tanh_grad)

        h_tilde_minus_h = self.h_tilde_minus_h[0:curr_max_seq_length, 0:curr_num_sequences]
        np.subtract(tanh_out, hs, out=h_tilde_minus_h)

        z_prod_grad = self.z_prod_grad[0:curr_max_seq_length, 0:curr_num_sequences]  # (T, B, H)
        np.multiply(tanh_grad, sigmoid_z_out, out=z_prod_grad)

        h_minus_h_prod_grad = self.h_minus_h_prod_grad[0:curr_max_seq_length, 0:curr_num_sequences]
        np.multiply(sigmoid_z_grad, h_tilde_minus_h, out=h_minus_h_prod_grad)

        h_prod_grad = self.h_prod_grad[0:curr_max_seq_length, 0:curr_num_sequences]  # (T, B, H)
        np.multiply(sigmoid_r_grad, hs, out=h_prod_grad)

        if curr_max_seq_length > 0:
            # (T-1, B, H, H)
            rhr = np.empty((curr_max_seq_length - 1, curr_num_sequences, dim_h, dim_h), dtype=self._dtype)

            # arg 1, 3 non-batched: (T-1, H), with batch: (T-1, B, H)
            # arg 2 is part of model and stays the same (H, H)
            # arg 4 rhr non-batched: (T-1, B, H, H)
            self.vect_element_wise_matrix_plus_diag(
                self.h_prod_grad[1:curr_max_seq_length, 0:curr_num_sequences, :],
                self.u[dim_h:(2 * dim_h)],
                self.sigmoid_r_out[1:curr_max_seq_length, 0:curr_num_sequences],
                rhr)

            self.__back_propagation_loop(delta_upper, 0, curr_max_seq_length, rhr)

        u_h = self.u[(2 * dim_h):, ]

        # fill dh2
        dh2 = self.dh2[0:curr_max_seq_length, 0:curr_num_sequences]
        dh = self.dh[0:curr_max_seq_length, 0:curr_num_sequences]
        np.multiply(dh, z_prod_grad, out=dh2[:, :, (2 * dim_h):])  # (T, B, H)
        np.multiply(dh, h_minus_h_prod_grad, out=dh2[:, :, 0:dim_h])  # (T, B, H)

        # it is faster to allocate a (B, H, H) here and do the element-wise multiplications inside the loop rather
        # than allocating a (T, B, H, H) and do the element-wise multiplications once outside the loop
        buf_b_hh = np.empty((curr_num_sequences, dim_h, dim_h), dtype=self._dtype)
        for t in range(curr_max_seq_length):
            # non-batch: (H1, 1) hadamard (H1, H2)
            # batched:   (B, H1, 1) hadamard (B, H1, H2)
            np.multiply(np.expand_dims(z_prod_grad[t], 2), u_h, out=buf_b_hh)
            # non-batch: (H2, ) hadamard (H1, H2) -> (H1, H2) hadamard (H1, H2) row vector repeated
            # batched:   (B, 1, H2) hadamard (B, H1, H2)
            np.multiply(np.expand_dims(h_prod_grad[t], 1), buf_b_hh, out=buf_b_hh)
            # non-batch: (H1, ) x (H1, H2) = (H2, )
            # batched:   B times of (H1, ) x (H1, H2) -> B times of (H2, )
            for b in range(curr_num_sequences):
                np.dot(dh[t, b], buf_b_hh[b], out=dh2[t, b, dim_h:(2 * dim_h)])

        if self.asserts_on:
            self._validate_zero_padding(dh2, max_seq_length=curr_max_seq_length)

        # reduce_sum (T, B, H) to (H, )
        np.sum(dh2, axis=(0, 1), out=self.db)

        # we can't easily sum over the T, B dimensions using matrix multiplications, so we use a loop for the first
        # dimension only (time)
        self.dw.fill(0.0)
        self.du.fill(0.0)
        buf_3hxd = np.empty((3 * dim_h, self.dim_d), dtype=self._dtype)
        buf_2hxh = np.empty((2 * dim_h, self.dim_h), dtype=self._dtype)
        for t in range(curr_max_seq_length):
            # (3*H, D) = (3*H, B) x (B, D) is the sum of outer products (3*H, 1) x (1, D) over the B sequences at time t
            np.dot(dh2[t].T, self.x[t], out=buf_3hxd)  # (3*H, B) x (B, D)
            self.dw += buf_3hxd
            np.dot(dh2[t, :, 0:(2 * dim_h)].T, hs[t], out=buf_2hxh)  # (2*H, B) x (B, H)
            self.du[:(2 * dim_h)] += buf_2hxh
            np.dot(dh2[t, :, (2 * dim_h):].T, self.r_prod_h[t, 0:curr_num_sequences], out=buf_2hxh[0:dim_h])
            self.du[(2 * dim_h):] += buf_2hxh[0:dim_h]

        # non-batch was:  (T, 3 * H) x (3 * H, D) = (T, D)
        # now with batch: (T, B, 3 * H) x (3 * H, D) = (T, B, D)
        # it is easier to just allocate delta_err each time instead of trying to reuse it, which is not supported by
        # np.dot(out=delta_err) when delta_err needs to be trimmed in the two leading dimensions
        delta_err = np.empty((self._curr_seq_length_dim_max, self._curr_num_sequences, self.dim_d), dtype=self._dtype)
        np.dot(dh2, self.w, out=delta_err[0:curr_max_seq_length])
        if curr_max_seq_length < self._curr_seq_length_dim_max:
            # required by base class contract
            delta_err[curr_max_seq_length:].fill(0.0)

        return delta_err

    def __back_propagation_loop(self, delta_upper, low_t, high_t, rhr):
        """
        Reverse iteration starting from high_t - 1, finishing at low_t, both inclusive.
        Populates self.dh[low_t:high_t]
        Args:
            delta_upper: error signal from upper layer, shape (L, B, H)
            low_t: low index, inclusive
            high_t: high index, exclusive
            rhr: produced by vect_element_wise_matrix_plus_diag(.), shape (L-1, B, H, H)
        """
        # dh_next[t] is my delta(t,T) x DH_{t} computed recursively by (13)
        # dh[t] is my delta(t, T)
        # dht[t] is my DH_{t}
        dim_h = self.dim_h
        curr_num_sequences = self._curr_num_sequences
        buf_b_h = self.buf_b_h[0:curr_num_sequences]
        buf_b_h_2 = np.empty((curr_num_sequences, dim_h), dtype=self._dtype)
        buf_hh = np.empty((dim_h, dim_h), dtype=self._dtype)
        u_z = self.u[0:dim_h]
        u_h = self.u[(2 * dim_h):]
        dh_next = self.dh_next[0:curr_num_sequences]  # (B, H)
        self.dh[high_t - 1, 0:curr_num_sequences] = delta_upper[high_t - 1]  # (B, H)
        h_minus_h_prod_grad = self.h_minus_h_prod_grad[:, 0:curr_num_sequences]
        one_minus_z = self.one_minus_z[:, 0:curr_num_sequences]
        z_prod_grad = self.z_prod_grad[:, 0:curr_num_sequences]
        # we could do here: np.dot(u_h, rhr) which is (H, H1) x (T-1, B, H1, H2) = (H, T-1, B, H2),
        # but for quick access we want (T-1, B, H, H2) and transposing or slicing was found to be too expensive
        for t in range(high_t - 2, low_t - 1, -1):
            dh = self.dh[t + 1, 0:curr_num_sequences]  # (B, H)
            np.multiply(dh, h_minus_h_prod_grad[t + 1], out=buf_b_h)
            np.dot(buf_b_h, u_z, out=dh_next)  # (B, H) x (H, H) => (B, H)
            np.multiply(dh, one_minus_z[t + 1], out=buf_b_h)  # (B, H)
            dh_next += buf_b_h
            np.multiply(dh, z_prod_grad[t + 1], out=buf_b_h)  # (B, H)
            # with no batch  rhr: (T-1, H, H)
            # mult: (H, H1) x (H1, H2) = (H, H2),  mult: (H, ) x (H, H2)
            # now with batch rhr: (T-1, B, H, H)
            # mult: (H, H1) with (B, H1, H2) -> (B, H, H2),  mult: (B, H, ) with (B, H, H2), so we have to have loop
            for b in range(curr_num_sequences):
                np.dot(u_h, rhr[t, b], out=buf_hh)
                np.dot(buf_b_h[b], buf_hh, out=buf_b_h_2[b])
            dh_next += buf_b_h_2  # (B, H)
            np.add(delta_upper[t], dh_next, out=self.dh[t, 0:curr_num_sequences])

    def __unpack_model_or_grad(self, params):
        hxh = self.dim_h * self.dim_h
        hxd = self.dim_h * self.dim_d
        of1 = 0
        of2 = 3 * hxd
        w = np.reshape(params[of1:of2], (3 * self.dim_h, self.dim_d))
        of1 = of2
        of2 += 3 * hxh
        u = np.reshape(params[of1:of2], (3 * self.dim_h, self.dim_h))
        of1 = of2
        of2 += 3 * self.dim_h
        b = np.reshape(params[of1:of2], (3 * self.dim_h, ))

        if self.asserts_on:
            # verify no memory copy
            # http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
            w.view().shape = (3 * hxd, )
            u.view().shape = (3 * hxh, )
            b.view().shape = (3 * self.dim_h, )
            assert np.shares_memory(w, params)

        return w, u, b

    def _set_model_references_in_place(self):
        self.w, self.u, self.b = self.__unpack_model_or_grad(self._model)

    def _set_gradient_references_in_place(self):
        self.dw, self.du, self.db = self.__unpack_model_or_grad(self._grad)

    def get_built_model(self):
        return np.concatenate((self.w.flatten(), self.u.flatten(), self.b))

    def get_built_gradient(self):
        return np.concatenate((self.dw.flatten(), self.du.flatten(), self.db))
