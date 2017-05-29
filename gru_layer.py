import numpy as np

import activation as ac
from neural_base import ComponentNN, glorot_init


class GruLayer(ComponentNN):
    def __init__(self, dim_d, dim_h, max_seq_length, dtype, asserts_on=True):
        self.dim_d, self.dim_h = dim_d, dim_h
        num_p = 3 * self.dim_h * self.dim_d + 3 * self.dim_h * self.dim_h + 3 * self.dim_h
        super(GruLayer, self).__init__(num_p, dtype)
        self.asserts_on = asserts_on
        self._seq_length = 0  # must be 0 before the first iteration
        self._max_seq_length = max_seq_length
        self.w = self.u = self.b = None
        self.dw = self.du = self.db = None
        # input to activation functions, dimensionality: (L, 3*H)
        self.act_in = np.empty((self._max_seq_length, 3 * self.dim_h), dtype=self._dtype)
        # output from activation functions, dimensionality: (L, 3*H)
        self.act_out = np.empty((self._max_seq_length, 3 * self.dim_h), dtype=self._dtype)
        # convenience view of last part of self.act_out
        self.hs_tilde = self.act_out[:, (2 * self.dim_h):]
        self.dhs_tilde_dh = np.empty((self._max_seq_length - 1, self.dim_h, self.dim_h), dtype=self._dtype)
        # self.wb_partial = np.empty((self._max_seq_length, 3 * self.dim_h), dtype=self._dtype)
        self.u_partial = np.empty(3 * self.dim_h, dtype=self._dtype)
        self.r_prod_h = np.empty((self._max_seq_length, self.dim_h), dtype=self._dtype)
        # hs[t, :] contains the hidden state for the (t-1)-input element, h[1] is first input hidden state
        # hs[0, :] contains the last hidden state of the previous sequence
        self.hs = np.empty((self._max_seq_length + 1, self.dim_h), dtype=dtype)
        self.ac_grad = np.empty((self._max_seq_length, 3 * self.dim_h), dtype=self._dtype)
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
        d.update({"dim_d": self.dim_d, "dim_h": self.dim_h, "max_seq_length": self._max_seq_length })
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
        self.b.fill(0.0)

    def model_glorot_init(self):
        assert self._model is not None
        for w in (self.w[:self.dim_h], self.w[self.dim_h:(2*self.dim_h)], self.w[(2*self.dim_h):]):
            np.copyto(w, glorot_init((self.dim_h, self.dim_d)).astype(self._dtype))
        for u in (self.u[:self.dim_h], self.u[self.dim_h:(2*self.dim_h)], self.u[(2*self.dim_h):]):
            np.copyto(u, glorot_init((self.dim_h, self.dim_h)).astype(self._dtype))
        self.b.fill(0.0)

    def forward_batch_debug(self, x):
        """ More readable, less optimized version than forward() which should be faster.

        It turns out that its running time is very similar to or marginally slower than the more "optimized" version.
        The optimized avoids memory allocations at the expense of more python function invocations.
        """
        if self.asserts_on:
            assert x.ndim == 2
            assert x.shape[1] == self.dim_d
            assert x.shape[0] <= self._max_seq_length

        # restore the last hidden state of the previous sequence (or what was set to via set_init_h())
        self.hs[0] = self.hs[self._seq_length]  # makes a copy, which is desirable

        self._seq_length = x.shape[0]
        self.x = x

        act_in = self.act_in[0:self._seq_length]
        act_out = self.act_out[0:self._seq_length]
        dim_h = self.dim_h

        w_zr, w_h = self.w[0:(2 * dim_h)], self.w[(2*dim_h):]
        u_zr, u_h = self.u[0:(2 * dim_h)], self.u[(2 * dim_h):]
        wb_zr = np.dot(self.x, w_zr.T) + self.b[0:(2*dim_h)]
        wb_h = np.dot(self.x, w_h.T) + self.b[(2*dim_h):]

        for t in xrange(self._seq_length):
            # ((2 * H, H) x (H, )) returns (2 * H, )
            act_in[t, 0:(2 * dim_h)] = wb_zr[t] + np.dot(u_zr, self.hs[t])
            act_out[t, 0:(2 * dim_h)] = ac.sigmoid(act_in[t, 0:(2 * dim_h)])
            z_t = act_out[t, 0:dim_h]
            r_t = act_out[t, (1*dim_h):(2*dim_h)]
            self.one_minus_z[t] = np.subtract(1.0, z_t)
            self.r_prod_h[t] = r_t * self.hs[t]
            act_in[t, (2 * dim_h):] = wb_h[t] + np.dot(u_h, self.r_prod_h[t])
            act_out[t, (2 * dim_h):] = np.tanh(act_in[t, (2 * dim_h):])
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

        act_in = self.act_in[0:self._seq_length]
        act_out = self.act_out[0:self._seq_length]
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
        for t in xrange(self._seq_length):
            # ((2 * H, H) x (H, )) returns (2 * H, )
            np.dot(self.u[0:(2*dim_h)], self.hs[t], out=u_partial[0:(2*dim_h)])
            np.add(wb_partial[t, 0:(2*dim_h)], u_partial[0:(2*dim_h)], out=act_in[t, 0:(2*dim_h)])
            ac.sigmoid(act_in[t, 0:(2*dim_h)], out=act_out[t, 0:(2*dim_h)])
            # act_out[t, 0:dim_h] = np.ones((dim_h,), dtype=self._dtype)  # for disabling update gate z_t
            z_t = act_out[t, 0:dim_h]  # select (1, H) from (T, 3*H)
            # act_out[t, (1*dim_h):(2*dim_h)] = np.ones((dim_h,), dtype=self._dtype)  # for disabling reset gate r_t
            r_t = act_out[t, (1*dim_h):(2*dim_h)]
            np.subtract(1.0, z_t, out=self.one_minus_z[t])
            r_prod_h = self.r_prod_h[t]
            np.multiply(r_t, self.hs[t], out=r_prod_h)
            # ((H, H) x (H, )) returns (H, )
            np.dot(self.u[(2*dim_h):], r_prod_h, out=u_partial[(2 * dim_h):])
            np.add(wb_partial[t, (2*dim_h):], u_partial[(2*dim_h):], out=act_in[t, (2 * dim_h):])
            np.tanh(act_in[t, (2*dim_h):], out=self.hs_tilde[t])
            np.multiply(z_t, self.hs_tilde[t], out=self.hs[t + 1])
            np.multiply(self.hs[t], self.one_minus_z[t], out=buf_h)
            self.hs[t + 1] += buf_h

        self.y = self.hs[1:(self._seq_length+1)]
        return self.y

    def vect_element_wise_matrix(self, n_vec, mat, out):
        """
        For each of N items: Repeat column vector vec Z times and compute element-wise product of vec and mat
        (both have shape (D, Z))

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
        For each of N items: Repeat column vector n_vec[t] D2 times, compute element-wise product of vec and mat
        (both have shape (D1, D2), D1=D2), add n_rvec[t] vector to its diagonal

        Args:
            n_vec: N vectors of size D1 (to be repeated as column vectors D2 times)
            mat: matrix of shape (D1, D2)
            n_rvec: N vectors of size D to be added to the diagonal
            out: array of shape (N, D1, D2) to hold the results
        """
        if self.asserts_on:
            assert n_vec.ndim == 2 and mat.ndim == 2
            assert n_vec.shape[1] == mat.shape[0] and mat.shape[0] == mat.shape[1]
            assert n_rvec.shape == n_vec.shape
        n, dim = n_vec.shape
        for i in xrange(n):
            # by taking transposes first, following avoids re-shape of n_vec[i] from (D1, ) to (D1, 1), it is faster
            # broadcasting: (D2, D1) hadamard (D1, ) -> (D2, D1) hadamard (D2, D1) = (D2, D1)
            np.multiply(mat.T, n_vec[i], out=out[i].T)
            # efficient way to add to the diagonal
            out[i].flat[0::dim + 1] += n_rvec[i]

    def backwards(self, delta_upper):
        if self.asserts_on:
            assert delta_upper.shape == (self._seq_length, self.dim_h)

        seq_length = self._seq_length
        dim_h = self.dim_h

        ac.sigmoid_grad(self.act_out[0:seq_length, 0:(2 * dim_h)],
                        out=self.ac_grad[0:seq_length, 0:(2 * dim_h)])  # (T, 2 * H)
        ac.tanh_grad(self.act_out[0:seq_length, (2 * dim_h):],
                     out=self.ac_grad[0:seq_length, (2 * dim_h):])  # (T, H)

        np.subtract(self.hs_tilde[0:seq_length], self.hs[0:seq_length], out=self.h_tilde_minus_h[0:seq_length])

        z_prod_grad = self.z_prod_grad[0:seq_length]  # (T, H)
        np.multiply(self.ac_grad[0:seq_length, (2 * dim_h):], self.act_out[0:seq_length, 0:dim_h], out=z_prod_grad)

        h_minus_h_prod_grad = self.h_minus_h_prod_grad[0:seq_length]  # (T, H)
        np.multiply(self.ac_grad[0:seq_length, 0:dim_h], self.h_tilde_minus_h[0:seq_length], out=h_minus_h_prod_grad)

        h_prod_grad = self.h_prod_grad[0:seq_length]  # (T, H)
        np.multiply(self.ac_grad[0:seq_length, dim_h:(2 * dim_h)], self.hs[0:seq_length], out=h_prod_grad)

        rhr = self.rhr[0:(seq_length-1)]

        self.vect_element_wise_matrix_plus_diag(self.h_prod_grad[1:seq_length], self.u[dim_h:(2 * dim_h)],
                                                self.act_out[1:seq_length, (1 * dim_h):(2 * dim_h)], rhr)

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
        for t in xrange(seq_length):
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

        np.dot(dh2[:, 0:(2*dim_h)].T, self.hs[0:seq_length], out=self.du[:(2*dim_h)])  # (2*H, T) x (T, D)
        r_dot_h = self.r_prod_h[0:seq_length]
        np.dot(dh2[:, (2*dim_h):].T, r_dot_h, out=self.du[(2*dim_h):])  # (H, T) x (T, D)

        # (T, 3 * H) x (3 * H, D) = (T, D)
        delta_err = self.delta_err_large[0:seq_length]
        np.dot(dh2, self.w, out=delta_err)

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
        # dht[t] is my DH_{t}
        dim_h = self.dim_h
        buf_h = self.buf_h
        buf_h_2 = np.empty(dim_h, dtype=self._dtype)
        buf_hh = self.buf_hh
        u_z = self.u[0:dim_h]
        u_h = self.u[(2 * dim_h):]
        dh_next = self.dh_next
        self.dh[high_t - 1] = delta_upper[high_t - 1]
        for t in xrange(high_t - 2, low_t - 1, -1):
            dh = self.dh[t + 1]
            np.multiply(dh, self.h_minus_h_prod_grad[t + 1], out=buf_h)
            np.dot(buf_h, u_z, out=dh_next)
            np.multiply(dh, self.one_minus_z[t+1], out=buf_h)
            dh_next += buf_h
            np.multiply(dh, self.z_prod_grad[t + 1], out=buf_h)
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
    #     for t in xrange(seq_length - 1):
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
