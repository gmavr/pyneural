import numpy as np

from softmax import softmax_1d_opt, softmax_2d_opt, softmax_2d_no_norm
from neural_base import LossNN, glorot_init


class CRFLayer(LossNN):
    """ Top scalar CRF layer
    
    Efficient vectorized implementation.
    """

    def __init__(self, dim_k, dtype, asserts_on=True):
        self.dim_k = dim_k
        super(CRFLayer, self).__init__(self.dim_k * self.dim_k, dtype)
        self.a_trans = None
        self.d_a_trans = None
        self.p_hat = None
        self.data, self.labels = None, None
        self.prev_label = None
        self.delta_err = None
        # forward and reverse scores, shape (N, K)
        # self.s[t] (self.s_reverse) holds forward (reverse) scores for each of K classes at time t
        self.s = self.s_reverse = None, None
        # forward and reverse scores minus the scale term self.c_norm, shape (N, K)
        self.rs = self.rs_reverse = None, None
        self.ps = None  # partial forward scores, shape (N, K, K)
        self.rs_reverse_first, self.s_reverse_first = None, None  # debug only
        # scale term used for numerical stability at time t, shape (N, )
        self.c_norm = None
        self.asserts_on = asserts_on

    def get_display_dict(self):
        d = self._init_display_dict()
        d["dim_k"] = self.dim_k
        return d

    def model_init_with(self, trans_scores):
        assert self._model is not None
        assert trans_scores.shape == (self.dim_k, self.dim_k)
        np.copyto(self.a_trans, trans_scores.astype(self._dtype))

    def model_normal_init(self, sd):
        assert self._model is not None
        np.copyto(self.a_trans, sd * np.random.standard_normal((self.dim_k, self.dim_k)).astype(self._dtype))

    def model_glorot_init(self):
        assert self._model is not None
        np.copyto(self.a_trans, glorot_init((self.dim_k, self.dim_k)).astype(self._dtype))

    def _set_model_references_in_place(self):
        self.a_trans = np.reshape(self._model, (self.dim_k, self.dim_k))

    def _set_gradient_references_in_place(self):
        self.d_a_trans = np.reshape(self._grad, (self.dim_k, self.dim_k))

    def set_prev_label(self, prev_label):
        self.prev_label = prev_label

    def compute_trellis_debug(self, data):
        """ Test only. Validates the efficient trellis implementation.
        
        Contains both the efficient and a slower but far more readable implementation.
        Verifies that they agree.
        Used for validation and model debugging.
        
        Args:
            data: input data, numpy array of shape (N, K)
        Raises:
            AssertionError: on illegal input and consistency errors
        """
        assert data.ndim == 2 and data.shape[1] == self.dim_k
        assert data.dtype == self._dtype
        assert data.shape[0] >= 1
        assert self.prev_label is not None

        num_samples = data.shape[0]
        self.data = data

        tolerance = 1e-7 if self._dtype == np.float32 else 1e-12

        # forward sequence

        self.s = np.empty((num_samples, self.dim_k), dtype=self._dtype)
        self.rs = np.empty((num_samples, self.dim_k), dtype=self._dtype)
        self.c_norm = np.empty((num_samples, ), dtype=self._dtype)
        self.ps = np.empty((num_samples, self.dim_k, self.dim_k), dtype=self._dtype)

        s0_1 = np.empty((self.dim_k, ), dtype=self._dtype)
        for j in xrange(self.dim_k):
            s0_1[j] = self.a_trans[self.prev_label, j] + data[0, j]

        self.s[0] = self.a_trans[self.prev_label] + data[0]  # (K, )
        self.c_norm[0] = np.max(self.s[0])
        self.ps[0, self.prev_label] = np.copy(self.s[0])  # important to copy, self.ps will be modified in back-prop
        self.rs[0] = self.s[0] - self.c_norm[0]  # not useful, for completeness only

        assert np.array_equal(self.s[0], s0_1)

        for t in xrange(1, num_samples):
            tmp1 = np.empty((self.dim_k, self.dim_k), dtype=self._dtype)
            exp_s = np.empty((self.dim_k, ), dtype=self._dtype)
            for j in xrange(self.dim_k):
                tmp1[j] = self.s[t - 1] + self.a_trans[:, j] + data[t, j]  # (K, )
                exp_s[j] = np.sum(np.exp(tmp1[j]))  # sum over i=1..K

            # broadcasting: (K, 1) + (K, K) + (K, ) -> (K, 1) + (K, K) + (1, K)
            # first column vector is replicated K times, self.data[t] is treated as row vector and replicated K times
            s_reshaped = np.reshape(self.s[t - 1], (self.dim_k, 1))
            self.ps[t] = s_reshaped + self.a_trans + self.data[t]

            assert np.array_equal(self.ps[t], tmp1.T)

            # self.s[t], self.rs[t] = self.log_sum_exp(self.ps[t], axis=0)
            # self.c_norm[t] = self.s[t, 0] - self.rs[t, 0]
            self.c_norm[t] = self.log_sum_exp_opt(self.ps[t], axis=0, out_rs_vector=self.rs[t], out_s_vector=self.s[t])
            s1 = np.log(exp_s)
            assert np.allclose(s1.T, self.s[t], rtol=tolerance)

        # backward sequence

        if num_samples == 1:
            self.rs_reverse = np.zeros((1, self.dim_k), dtype=self._dtype)
            return

        self.s_reverse = np.empty((num_samples - 1, self.dim_k), dtype=self._dtype)
        self.rs_reverse = np.empty((num_samples, self.dim_k), dtype=self._dtype)
        # self.rs_reverse[num_samples-1] needs to exist for dimension compatibility in backpropagation
        self.rs_reverse[-1] = np.zeros((1, self.dim_k), dtype=self._dtype)

        t = num_samples - 1
        tmp1 = np.empty((self.dim_k, self.dim_k), dtype=self._dtype)
        exp_s_reverse = np.empty((self.dim_k, ), dtype=self._dtype)
        for j in xrange(self.dim_k):
            tmp1[j] = self.a_trans[j] + data[t]  # (K, )
            exp_s_reverse[j] = np.sum(np.exp(tmp1[j]))  # sum over i=1..K

        # broadcasting of (K, K) + (K, ) -> (K, K) + (1, K): the row vector data[t] is replicated K times
        prs = self.a_trans + data[t]  # (K, K)
        assert np.array_equal(prs, tmp1)

        # numerically stable version not really compelling for first iteration, but easier to just do it
        self.log_sum_exp_opt(prs, axis=1, out_rs_vector=self.rs_reverse[t - 1], out_s_vector=self.s_reverse[t - 1])

        s_reverse = np.log(exp_s_reverse)  # (1, K)
        assert np.allclose(s_reverse, self.s_reverse[t - 1], rtol=tolerance)

        # s_reverse[num_samples - 2, j] == S'_(N-1, j). It holds S'_(N, j) == 0 for all j and therefore omitted in code

        for t in xrange(num_samples - 2, 0, -1):
            tmp1 = np.empty((self.dim_k, self.dim_k), dtype=self._dtype)
            exp_s_reverse = np.empty((self.dim_k, ), dtype=self._dtype)
            for j in xrange(self.dim_k):
                tmp1[j] = self.a_trans[j] + data[t] + self.s_reverse[t]  # (K, )
                exp_s_reverse[j] = np.sum(np.exp(tmp1[j]))  # sum over i=1..K

            # broadcasting of (K, K) + (K, ) + (K, ):
            # the 2 row vectors are replicated K times
            prs = self.a_trans + data[t] + self.s_reverse[t]  # (K, K)
            assert np.array_equal(prs, tmp1)

            s_reverse = np.log(exp_s_reverse)
            # self.s_reverse[t - 1], self.rs_reverse[t - 1] = self.log_sum_exp(prs, 1)
            self.log_sum_exp_opt(prs, axis=1, out_rs_vector=self.rs_reverse[t - 1], out_s_vector=self.s_reverse[t - 1])
            assert np.allclose(s_reverse, self.s_reverse[t - 1], rtol=tolerance)

    def compute_trellis(self, data):
        if self.asserts_on:
            assert data.ndim == 2 and data.shape[1] == self.dim_k
            assert data.dtype == self._dtype
            assert data.shape[0] >= 1
            assert self.prev_label is not None

        num_samples = data.shape[0]
        self.data = data

        # forward sequence

        self.s = np.empty((num_samples, self.dim_k), dtype=self._dtype)
        self.rs = np.empty((num_samples, self.dim_k), dtype=self._dtype)
        self.c_norm = np.empty((num_samples, ), dtype=self._dtype)
        self.ps = np.empty((num_samples, self.dim_k, self.dim_k), dtype=self._dtype)

        self.s[0] = self.a_trans[self.prev_label] + data[0]  # (K, )
        self.c_norm[0] = np.max(self.s[0])
        self.ps[0, self.prev_label] = np.copy(self.s[0])  # important to copy, self.ps will be modified in back-prop
        self.rs[0] = self.s[0] - self.c_norm[0]  # not useful, for completeness only

        # This recursively updates quantities at time t from their values at time t-1, therefore it cannot vectorized.
        # A loop is necessary.
        for t in xrange(1, num_samples):
            # broadcasting: (K, 1) + (K, K) + (K, ) -> (K, 1) + (K, K) + (1, K)
            # first column vector is replicated K times, self.data[t] is treated as row vector and replicated K times
            s_reshaped = np.reshape(self.s[t - 1], (self.dim_k, 1))
            self.ps[t] = s_reshaped + self.a_trans + self.data[t]
            # self.s[t], self.rs[t] = self.log_sum_exp(self.ps[t], axis=0)
            # self.c_norm[t] = self.s[t, 0] - self.rs[t, 0]
            self.c_norm[t] = self.log_sum_exp_opt(self.ps[t], axis=0, out_rs_vector=self.rs[t], out_s_vector=self.s[t])

        # backward sequence

        if num_samples == 1:
            self.rs_reverse = np.zeros((1, self.dim_k), dtype=self._dtype)
            return

        self.s_reverse = np.empty((num_samples - 1, self.dim_k), dtype=self._dtype)
        self.rs_reverse = np.empty((num_samples, self.dim_k), dtype=self._dtype)
        # self.rs_reverse[-1] needs to exist for dimension compatibility in backpropagation
        self.rs_reverse[-1] = np.zeros((1, self.dim_k), dtype=self._dtype)

        t = num_samples - 1
        # broadcasting of (K, K) + (K, ) -> (K, K) + (1, K): the row vector data[t] is replicated K times
        prs = self.a_trans + data[t]  # (K, K)

        # numerically stable version not really compelling for first iteration, but easier to just do it
        self.log_sum_exp_opt(prs, axis=1, out_rs_vector=self.rs_reverse[t - 1], out_s_vector=self.s_reverse[t - 1])

        # s_reverse[num_samples - 2, j] == S'_(N-1, j). It holds S'_(N, j) == 0 for all j and therefore omitted in code

        # This recursively updates quantities at time t from their values at time t-1, therefore it cannot vectorized.
        # A loop is necessary.
        for t in xrange(num_samples - 2, 0, -1):
            # broadcasting of (K, K) + (K, ) + (K, ):
            # the 2 row vectors are replicated K times
            prs = self.a_trans + data[t] + self.s_reverse[t]  # (K, K)
            # self.s_reverse[t - 1], self.rs_reverse[t - 1] = self.log_sum_exp(prs, 1)
            self.log_sum_exp_opt(prs, axis=1, out_rs_vector=self.rs_reverse[t - 1], out_s_vector=self.s_reverse[t - 1])

    @staticmethod
    def log_sum_exp(s_array, axis):
        """ Computes log sum of 2-dim array along given dimension in a numerically stable fashion
        
        Args:
            s_array: current model parameters, np.array of shape (K, K)
            axis: dimension across which summation occurs
        Returns:
            s_vector: log sum 
            rs_vector: log sum excluding the scale offset number
        """
        # assert s_array.shape == (self.dim_k, self.dim_k)

        # numerical stability trick: Because x_i can be potentially huge numbers, we subtract from all the same number.
        # log(sum_i(exp(x_i))) = log(sum_i(exp(x_i-y+y))) = log(exp(y)*sum_i(exp(x_i-y))) = y + sum_i(exp(x_i-y)))

        scale_term = np.max(s_array)
        rs_vector = np.log(np.sum(np.exp(s_array - scale_term), axis=axis))
        s_vector = scale_term + rs_vector

        return s_vector, rs_vector

    @staticmethod
    def log_sum_exp_opt(s_array, axis, out_rs_vector, out_s_vector):
        """ Same as log_sum_exp, but output is in-place of supplied vector holders. """
        # numerical stability trick: Because x_i can be potentially huge numbers, we subtract from all the same number.
        # log(sum_i(exp(x_i))) = log(sum_i(exp(x_i-y+y))) = log(exp(y)*sum_i(exp(x_i-y))) = y + sum_i(exp(x_i-y)))

        # following lines compute in-place:
        # rs_vector = np.log(np.sum(np.exp(s_array - scale_term), axis=axis))
        # s_vector = scale_term + rs_vector

        scale_term = np.max(s_array)

        z1 = s_array - scale_term
        np.exp(z1, out=z1)
        np.sum(z1, axis=axis, out=out_rs_vector)
        np.log(out_rs_vector, out=out_rs_vector)

        np.add(out_rs_vector, scale_term, out=out_s_vector)

        return scale_term

    def _validate_labels(self, labels):
        assert labels.ndim == 1
        assert len(self.data) == len(labels)
        # assert labels.dtype == np.int
        assert np.max(labels) < self.dim_k and 0 <= np.min(labels)

    def score_for_seq(self, labels):
        if self.asserts_on:
            self._validate_labels(labels)

        num_samples = self.data.shape[0]

        score = self.a_trans[self.prev_label, labels[0]]
        score += np.sum(self.a_trans[labels[0:(num_samples-1)], labels[1:num_samples]])
        score += np.sum(self.data[xrange(0, num_samples), labels[0:num_samples]])
        # The above is "advanced array indexing" ("fancy indexing") with first coordinate a sequence object and second
        # an ndarray. The above is NOT equal to self.data[0:num_samples, labels[0:num_samples]]] which mixes "slicing"
        # and "advanced array indexing".
        # See: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
        return score

    def score_for_correct_seq(self):
        return self.score_for_seq(self.labels)

    def all_seq_log_sum_exp(self):
        num_samples = self.data.shape[0]
        return self.c_norm[num_samples-1] + np.log(np.sum(np.exp(self.rs[num_samples-1])))

    def all_seq_log_sum_exp_reverse(self):
        """ For testing only """
        num_samples = self.data.shape[0]
        if num_samples == 1:
            tmp = self.a_trans[self.prev_label] + self.data[0]
        else:
            tmp = self.a_trans[self.prev_label] + self.data[0] + self.s_reverse[0]
        max_tmp = np.max(tmp)
        self.rs_reverse_first = np.log(np.sum(np.exp(tmp - max_tmp)))
        self.s_reverse_first = max_tmp + self.rs_reverse_first
        return self.s_reverse_first

    def loss_func(self, labels):
        # - correct_sequence + all_sequences
        return -self.score_for_seq(labels) + self.all_seq_log_sum_exp()

    def loss_func_for_correct_seq(self):
        # - correct_sequence + all_sequences
        return -self.score_for_correct_seq() + self.all_seq_log_sum_exp()

    def get_most_probable_seq(self, class_dtype=np.int):
        """ Return most probable sequence using the Viterbi algorithm """
        num_samples = self.data.shape[0]
        path = np.empty((num_samples, ), dtype=class_dtype)
        scores = self.a_trans[self.prev_label] + self.data[0]
        path[0] = np.argmax(scores)
        score = scores[path[0]]

        for t in range(1, num_samples):
            scores = self.a_trans[path[t-1]] + self.data[t]
            path[t] = np.argmax(scores)
            score += scores[path[t]]
        return path

    def set_labels(self, labels):
        if self.asserts_on:
            self._validate_labels(labels)
        self.labels = labels

    def forward(self, data, labels):
        self._x, self._y_true = data, labels

        self.compute_trellis(data)
        self.set_labels(labels)
        return self.loss_func_for_correct_seq()

    def backwards_slower(self):
        num_samples = self.data.shape[0]

        ds = np.zeros((num_samples, self.dim_k))
        ds[xrange(num_samples), self.labels] = -1
        self.delta_err = ds + softmax_2d_no_norm(self.rs + self.rs_reverse)

        if num_samples > 1:
            self.ps[0, self.prev_label] += self.s_reverse[0]  # (K, )

            # t = num_samples - 1 excluded from the loop on purpose, self.ps[num_samples - 1] should not be modified
            for t in xrange(1, num_samples - 1):
                # broadcasting: (K, K) + (K, ) -> (K, K) + (1, K) row vector self.s_reverse[t] replicated K times
                self.ps[t] += self.s_reverse[t]

        # helper for partial derivative
        q = np.empty((num_samples, self.dim_k, self.dim_k), dtype=self._dtype)

        q[0] = np.zeros((self.dim_k, self.dim_k), dtype=self._dtype)
        # q[0, self.prev_label] = softmax_1d(self.ps[0, self.prev_label])
        softmax_1d_opt(self.ps[0, self.prev_label], out=q[0, self.prev_label])

        for t in xrange(1, num_samples):
             qtmp = softmax_1d_opt(np.reshape(self.ps[t], (self.dim_k * self.dim_k, )))
             q[t] = np.reshape(qtmp, (self.dim_k, self.dim_k))

        # element-wise addition of the N arrays of size (K, K)
        np.sum(q, axis=0, out=self.d_a_trans)

        self.d_a_trans[self.prev_label, self.labels[0]] -= 1
        for t in xrange(1, num_samples):
            self.d_a_trans[self.labels[t-1], self.labels[t]] -= 1

        # self.post_back_propagation_assert()

        return self.delta_err

    def backwards(self):
        num_samples = self.data.shape[0]

        self.delta_err = softmax_2d_no_norm(self.rs + self.rs_reverse)
        self.delta_err[xrange(num_samples), self.labels] -= 1

        if num_samples > 1:
            self.ps[0, self.prev_label] += self.s_reverse[0]  # (K, )

            # t = num_samples - 1 excluded from the loop on purpose, self.ps[num_samples - 1] should not be modified
            # add one dimension of size 1 for broadcasting to work
            self.ps[1:(num_samples - 1)] += np.expand_dims(self.s_reverse[1:(num_samples - 1)], axis=1)
            # above is equivalent to following but without the loop
            # for t in xrange(1, num_samples - 1):
            #     # broadcasting: (K, K) + (K, ) -> (K, K) + (1, K) row vector self.s_reverse[t] replicated K times
            #     self.ps[t] += self.s_reverse[t]

        # helper for partial derivative
        q = np.empty((num_samples, self.dim_k, self.dim_k), dtype=self._dtype)

        q[0] = np.zeros((self.dim_k, self.dim_k), dtype=self._dtype)
        softmax_1d_opt(self.ps[0, self.prev_label], out=q[0, self.prev_label])

        pst = np.reshape(self.ps[1:num_samples], (num_samples-1, self.dim_k * self.dim_k))
        q2 = np.reshape(q, (num_samples, self.dim_k * self.dim_k))
        softmax_2d_opt(pst, out=q2[1:num_samples])

        # element-wise addition of the N arrays of size (K, K)
        np.sum(q, axis=0, out=self.d_a_trans)

        self.d_a_trans[self.prev_label, self.labels[0]] -= 1
        # XXX figure out how to vectorize this loop
        for t in xrange(1, num_samples):
            self.d_a_trans[self.labels[t-1], self.labels[t]] -= 1

        # self.post_back_propagation_assert()

        return self.delta_err

    def post_backwards_assert(self):
        tolerance = 1e-6 if self._dtype == np.float32 else 1e-15

        num_samples = self.data.shape[0]

        all_seq_log_sum_exp = self.all_seq_log_sum_exp()

        tmp = np.log(np.sum(np.exp(self.ps[0][self.prev_label])))
        assert np.fabs(1 - tmp / all_seq_log_sum_exp) <= tolerance

        for t in range(1, num_samples):
            tmp = np.log(np.sum(np.exp(self.ps[t])))
            assert np.fabs(1 - tmp / all_seq_log_sum_exp) <= tolerance

    def forward_backwards(self, data, labels):
        loss = self.forward(data, labels)
        self.backwards()
        return loss, self._grad, self.delta_err

    def get_built_model(self):
        return self.a_trans.flatten()

    def get_built_gradient(self):
        return self.d_a_trans.flatten()

