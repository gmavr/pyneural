from typing import Tuple

import numpy as np

from pyneural.neural_base import LossNN, glorot_init
from pyneural.softmax import softmax_1d_opt, softmax_2d_opt, softmax_2d_no_norm


class CRFLayer(LossNN):
    """ Top scalar CRF layer

    Efficient vectorized implementation.
    """

    def __init__(self, dim_k, max_seq_length, dtype, asserts_on=True):
        """
        Args:
            dim_k: number of classes
            max_seq_length: maximum sequence length for any sequence that will be presented in forward and backwards.
            dtype: numpy type of all parameters and inputs: np.float32 or np.float64
            asserts_on: perform invariant and consistency assertions. Recommended to set to False only at final steps of
                training large models
        """
        self.dim_k = dim_k
        super(CRFLayer, self).__init__(self.dim_k * self.dim_k, dtype)
        self.max_seq_length = max_seq_length
        self.asserts_on = asserts_on
        self.num_samples_prev = -1
        self.a_trans = None  # transition scores, i.e. the model, shape (K, K)
        self.d_a_trans = None  # gradient of the model
        self.p_hat = None
        self.data, self.labels = None, None  # (N, K), (K, ) externally owned
        self.prev_label = None
        self.delta_err = None
        # self.s, self.s_reverse hold the trellis, the scores of paths in the log-domain. Shape: (N, K)
        # self.s[t] (self.s_reverse[t]) holds the forward (reverse) scores for each of K classes at time t.
        # Score self.s[t, k] is the log sum exp of scores of all possible (forward) paths starting from self.prev_label
        # at time ts = -1 and ending at label k at time t. (For t = 0, that is equal to self.a_trans[0, k] + data[0, k])
        self.s = np.empty((self.max_seq_length, self.dim_k), dtype=self._dtype)
        # Score self.s_reverse[t, k] is the log sum exp of scores of all possible (backward) paths starting from any
        # label at time ts = N - 1 and ending at label k at time t. (For t = N - 1, that is equal to 0 and is omitted).
        self.s_reverse = np.empty((self.max_seq_length - 1, self.dim_k), dtype=self._dtype)
        # scale term used for numerical stability at time N - 1 (no need to remember the scaling terms for other times)
        self.scale_term = None
        # self.rs[0:num_samples] and self.rs_reverse[0:num_samples] are computed in the forward pass and are only read
        # in the backward pass (back-propagation). self.rs[num_samples - 1] is also used by self.all_seq_log_sum_exp(.)
        # self.rs[t] = self.r[t] - scale_term[t], shape (N, K)
        self.rs = np.empty((self.max_seq_length, self.dim_k), dtype=self._dtype)
        # self.rs_reverse[t] = self.s_reverse[t] - scale_term[t], shape (N, K)
        self.rs_reverse = np.empty((self.max_seq_length, self.dim_k), dtype=self._dtype)
        # Partial forward scores. Shape (N, K, K)
        self.ps = None
        self.rs_reverse_first, self.s_reverse_first = None, None  # debug only
        # scratch buffer
        self.buf_k2 = np.empty((self.dim_k, self.dim_k), dtype=self._dtype)

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
        assert 1 <= data.shape[0] <= self.max_seq_length
        assert self.prev_label is not None

        num_samples = data.shape[0]
        self.data = data

        tolerance = 1e-7 if self._dtype == np.float32 else 1e-12

        # forward sequence

        # "trim" to proper size (no copy)
        s = self.s[0:num_samples]
        rs = self.rs[0:num_samples]
        ps = self.ps = np.empty((num_samples, self.dim_k, self.dim_k), dtype=self._dtype)

        s0_1 = np.empty((self.dim_k, ), dtype=self._dtype)
        for j in range(self.dim_k):
            s0_1[j] = self.a_trans[self.prev_label, j] + data[0, j]

        s[0] = self.a_trans[self.prev_label] + data[0]  # (K, )
        scale_term = np.max(s[0])
        # ps[0] except for ps[0, self.prev_label] is never used, do not bother initializing
        ps[0, self.prev_label] = np.copy(s[0])  # important to copy, ps will be modified in back-prop
        rs[0] = s[0] - scale_term

        assert np.array_equal(s[0], s0_1)

        for t in range(1, num_samples):
            tmp1 = np.empty((self.dim_k, self.dim_k), dtype=self._dtype)
            exp_s = np.empty((self.dim_k, ), dtype=self._dtype)
            for j in range(self.dim_k):
                tmp1[j] = s[t - 1] + self.a_trans[:, j] + data[t, j]  # (K, )
                exp_s[j] = np.sum(np.exp(tmp1[j]))  # sum over i=1..K

            # broadcasting: (K, 1) + (K, K) + (K, ) -> (K, 1) + (K, K) + (1, K)
            # first column vector is replicated K times, self.data[t] is treated as row vector and replicated K times
            s_reshaped = np.reshape(s[t - 1], (self.dim_k, 1))
            ps[t] = s_reshaped + self.a_trans + self.data[t]

            assert np.array_equal(ps[t], tmp1.T)

            # s[t], rs[t] = self.log_sum_exp(ps[t], axis=0)
            # scale_term = s[t, 0] - rs[t, 0]
            scale_term = self._log_sum_exp_opt(ps[t], axis=0, out_rs_vector=rs[t], out_s_vector=s[t])
            s1 = np.log(exp_s)
            assert np.allclose(s1.T, s[t], rtol=tolerance)

        self.scale_term = scale_term

        # backward sequence

        # "trim" to proper sizes (no copies)
        s_reverse = self.s_reverse[0:(num_samples - 1)]
        rs_reverse = self.rs_reverse[0:num_samples]

        # rs_reverse[-1] needs to exist for dimension compatibility in backpropagation
        rs_reverse[-1].fill(0.0)
        self.num_samples_prev = num_samples

        if num_samples == 1:
            return

        t = num_samples - 1
        tmp1 = np.empty((self.dim_k, self.dim_k), dtype=self._dtype)
        exp_s_reverse = np.empty((self.dim_k, ), dtype=self._dtype)
        for j in range(self.dim_k):
            tmp1[j] = self.a_trans[j] + data[t]  # (K, )
            exp_s_reverse[j] = np.sum(np.exp(tmp1[j]))  # sum over i=1..K

        # broadcasting of (K, K) + (K, ) -> (K, K) + (1, K): the row vector data[t] is replicated K times
        prs = self.a_trans + data[t]  # (K, K)
        assert np.array_equal(prs, tmp1)

        # numerically stable version not really compelling for first iteration, but easier to just do it
        self._log_sum_exp_opt(prs, axis=1, out_rs_vector=rs_reverse[t - 1], out_s_vector=s_reverse[t - 1])

        s_reverse_comp = np.log(exp_s_reverse)  # (1, K)
        assert np.allclose(s_reverse_comp, s_reverse[t - 1], rtol=tolerance)

        # s_reverse[num_samples - 2, j] == S'_(N-1, j). It holds S'_(N, j) == 0 for all j and therefore omitted in code

        for t in range(num_samples - 2, 0, -1):
            tmp1 = np.empty((self.dim_k, self.dim_k), dtype=self._dtype)
            exp_s_reverse = np.empty((self.dim_k, ), dtype=self._dtype)
            for j in range(self.dim_k):
                tmp1[j] = self.a_trans[j] + data[t] + s_reverse[t]  # (K, )
                exp_s_reverse[j] = np.sum(np.exp(tmp1[j]))  # sum over i=1..K

            # broadcasting of (K, K) + (K, ) + (K, ):
            # the 2 row vectors are replicated K times
            prs = self.a_trans + data[t] + s_reverse[t]  # (K, K)
            assert np.array_equal(prs, tmp1)

            s_reverse_comp = np.log(exp_s_reverse)
            # s_reverse[t - 1], rs_reverse[t - 1] = self.log_sum_exp(prs, 1)
            self._log_sum_exp_opt(prs, axis=1, out_rs_vector=rs_reverse[t - 1], out_s_vector=s_reverse[t - 1])
            assert np.allclose(s_reverse_comp, s_reverse[t - 1], rtol=tolerance)

    def compute_trellis(self, data):
        if self.asserts_on:
            assert data.ndim == 2 and data.shape[1] == self.dim_k
            assert data.dtype == self._dtype
            assert 1 <= data.shape[0] <= self.max_seq_length
            assert self.prev_label is not None

        num_samples = data.shape[0]
        self.data = data

        # forward sequence

        # "trim" to proper sizes (no copies)
        s = self.s[0:num_samples]
        rs = self.rs[0:num_samples]
        if self.num_samples_prev != num_samples:
            self.ps = np.empty((num_samples, self.dim_k, self.dim_k), dtype=self._dtype)
        ps = self.ps

        s[0] = self.a_trans[self.prev_label] + data[0]  # (K, )
        scale_term = np.max(s[0])
        # ps[0] except for ps[0, self.prev_label] is never used, do not bother initializing
        ps[0, self.prev_label] = np.copy(s[0])  # important to copy, ps will be modified in back-prop
        rs[0] = s[0] - scale_term

        # This recursively updates quantities at time t from their values at time t-1, therefore it cannot vectorized.
        # A loop is necessary.
        for t in range(1, num_samples):
            # broadcasting: (K, 1) + (K, K) + (K, ) -> (K, 1) + (K, K) + (1, K)
            # first column vector is replicated K times, self.data[t] is treated as row vector and replicated K times
            s_reshaped = np.reshape(s[t - 1], (self.dim_k, 1))
            # ps[t] = s_reshaped + self.a_trans + self.data[t]
            np.add(s_reshaped, self.a_trans, out=ps[t])
            np.add(ps[t], self.data[t], out=ps[t])
            # s[t], rs[t] = self.log_sum_exp(ps[t], axis=0)
            # scale_term = s[t, 0] - rs[t, 0]
            scale_term = self._log_sum_exp_opt(ps[t], axis=0, out_rs_vector=rs[t], out_s_vector=s[t])

        self.scale_term = scale_term

        # backward sequence

        # "trim" to proper sizes (no copies)
        s_reverse = self.s_reverse[0:(num_samples - 1)]
        rs_reverse = self.rs_reverse[0:num_samples]

        # rs_reverse[-1] needs to exist for dimension compatibility in backpropagation
        rs_reverse[-1].fill(0.0)
        self.num_samples_prev = num_samples

        if num_samples == 1:
            return

        prs = self.buf_k2
        t = num_samples - 1
        # broadcasting of (K, K) + (K, ) -> (K, K) + (1, K): the row vector data[t] is replicated K times
        # prs = self.a_trans + data[t]  # (K, K)
        np.add(self.a_trans, data[t], out=prs)

        # numerically stable version not really compelling for first iteration, but easier to just do it
        self._log_sum_exp_opt(prs, axis=1, out_rs_vector=rs_reverse[t - 1], out_s_vector=s_reverse[t - 1])

        # s_reverse[num_samples - 2, j] == S'_(N-1, j). It holds S'_(N, j) == 0 for all j and therefore omitted in code

        # This recursively updates quantities at time t from their values at time t-1, therefore it cannot vectorized.
        # A loop is necessary.
        for t in range(num_samples - 2, 0, -1):
            # broadcasting of (K, K) + (K, ) + (K, ):
            # the 2 row vectors are replicated K times
            # prs = self.a_trans + data[t] + s_reverse[t]  # (K, K)
            np.add(self.a_trans, data[t], out=prs)
            np.add(prs, s_reverse[t], out=prs)
            # s_reverse[t - 1], rs_reverse[t - 1] = self.log_sum_exp(prs, 1)
            self._log_sum_exp_opt(prs, axis=1, out_rs_vector=rs_reverse[t - 1], out_s_vector=s_reverse[t - 1])

    @staticmethod
    def _log_sum_exp(s_array: np.array, axis: int) -> Tuple[np.array, np.array]:
        """ Computes log sum exp of 2-dim array along given dimension in a numerically stable fashion

        Args:
            s_array: np.array of shape (K, K)
            axis: dimension across which summation occurs
        Returns:
            s_vector: np.array of shape (K, ), log sum exp
            rs_vector: np.array of shape (K, ), log sum exp minus the scaling term
        """
        # assert s_array.shape == (self.dim_k, self.dim_k)

        # numerical stability trick: Because x_i can be potentially huge numbers, we subtract their max from all.
        # log(sum_i(exp(x_i))) = log(sum_i(exp(x_i-y+y))) = log(exp(y)*sum_i(exp(x_i-y))) = y + sum_i(exp(x_i-y)))

        scale_term = np.max(s_array)
        rs_vector = np.log(np.sum(np.exp(s_array - scale_term), axis=axis))
        s_vector = scale_term + rs_vector

        return s_vector, rs_vector

    def _log_sum_exp_opt(self, s_array: np.array, axis: int, out_rs_vector: np.array, out_s_vector: np.array) -> float:
        """ Same as _log_sum_exp(s_array, axis), but output is in-place of supplied vector holders.
        It also returns the  numerical stability scaling term it chooses.
        """

        if self.asserts_on:
            assert s_array.shape == (self.dim_k, self.dim_k)

        # following lines compute in-place:
        # rs_vector = np.log(np.sum(np.exp(s_array - scale_term), axis=axis))
        # s_vector = scale_term + rs_vector

        scale_term = np.max(s_array)

        # in-place: self.buf_k2 = s_array - scale_term
        buf_k2 = self.buf_k2
        np.subtract(s_array, scale_term, out=buf_k2)
        np.exp(buf_k2, out=buf_k2)
        np.sum(buf_k2, axis=axis, out=out_rs_vector)
        np.log(out_rs_vector, out=out_rs_vector)

        np.add(out_rs_vector, scale_term, out=out_s_vector)

        return scale_term

    def _validate_labels(self, labels):
        assert labels.ndim == 1
        assert len(self.data) == len(labels)
        assert np.issubdtype(labels.dtype, np.integer)
        assert np.max(labels) < self.dim_k and 0 <= np.min(labels)

    def score_for_seq(self, labels) -> float:
        if self.asserts_on:
            self._validate_labels(labels)

        num_samples = self.data.shape[0]

        score = self.a_trans[self.prev_label, labels[0]]
        score += np.sum(self.a_trans[labels[0:(num_samples-1)], labels[1:num_samples]])
        score += np.sum(self.data[range(0, num_samples), labels[0:num_samples]])
        # The above is "advanced array indexing" ("fancy indexing") with first coordinate a sequence object and second
        # an ndarray. The above is NOT equal to self.data[0:num_samples, labels[0:num_samples]]] which mixes "slicing"
        # and "advanced array indexing".
        # See: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
        return score

    def all_seq_log_sum_exp(self) -> float:
        """Return the log sum exp of the scores of all possible sequences.
        Can be called after the trellis has been constructed.
        """
        num_samples = self.data.shape[0]
        return self.scale_term + np.log(np.sum(np.exp(self.rs[num_samples-1])))

    def all_seq_log_sum_exp_reverse(self) -> float:
        """Returns the same as all_seq_log_sum_exp(.) except that it operates on the reverse path.
        Primarily for testing.
        """
        num_samples = self.data.shape[0]
        if num_samples == 1:
            tmp = self.a_trans[self.prev_label] + self.data[0]
        else:
            tmp = self.a_trans[self.prev_label] + self.data[0] + self.s_reverse[0]
        max_tmp = np.max(tmp)
        self.rs_reverse_first = np.log(np.sum(np.exp(tmp - max_tmp)))
        self.s_reverse_first = max_tmp + self.rs_reverse_first
        return self.s_reverse_first

    def loss_for_seq(self, labels) -> float:
        # - given_label_sequence + all_sequences
        return -self.score_for_seq(labels) + self.all_seq_log_sum_exp()

    def set_labels(self, labels):
        if self.asserts_on:
            self._validate_labels(labels)
        self.labels = labels

    def forward(self, data, labels):
        self._x, self._y_true = data, labels

        self.compute_trellis(data)
        self.set_labels(labels)

        # - correct_label_sequence + all_sequences
        return -self.score_for_seq(self.labels) + self.all_seq_log_sum_exp()

    def backwards_slower(self):
        num_samples = self.data.shape[0]

        ds = np.zeros((num_samples, self.dim_k))
        ds[range(num_samples), self.labels] = -1
        self.delta_err = ds + softmax_2d_no_norm(self.rs[0:num_samples] + self.rs_reverse[0:num_samples])

        ps = self.ps

        if num_samples > 1:
            ps[0, self.prev_label] += self.s_reverse[0]  # (K, )

            # t = num_samples - 1 excluded from the loop on purpose, ps[num_samples - 1] should not be modified
            for t in range(1, num_samples - 1):
                # broadcasting: (K, K) + (K, ) -> (K, K) + (1, K) row vector self.s_reverse[t] replicated K times
                ps[t] += self.s_reverse[t]

        # helper for partial derivative
        q = np.empty((num_samples, self.dim_k, self.dim_k), dtype=self._dtype)

        q[0].fill(0.0)
        # q[0, self.prev_label] = softmax_1d(ps[0, self.prev_label])
        softmax_1d_opt(ps[0, self.prev_label], out=q[0, self.prev_label])

        for t in range(1, num_samples):
            qtmp = softmax_1d_opt(np.reshape(ps[t], (self.dim_k * self.dim_k, )))
            q[t] = np.reshape(qtmp, (self.dim_k, self.dim_k))

        # element-wise addition of the N arrays of size (K, K)
        np.sum(q, axis=0, out=self.d_a_trans)

        self.d_a_trans[self.prev_label, self.labels[0]] -= 1
        for t in range(1, num_samples):
            self.d_a_trans[self.labels[t-1], self.labels[t]] -= 1

        # self.post_back_propagation_assert()

        return self.delta_err

    def backwards(self):
        num_samples = self.data.shape[0]

        self.delta_err = softmax_2d_no_norm(self.rs[0:num_samples] + self.rs_reverse[0:num_samples])
        self.delta_err[range(num_samples), self.labels] -= 1

        ps = self.ps

        if num_samples > 1:
            ps[0, self.prev_label] += self.s_reverse[0]  # (K, )

            # t = num_samples - 1 excluded from the loop on purpose, ps[num_samples - 1] should not be modified
            # add one dimension of size 1 for broadcasting to work
            ps[1:(num_samples - 1)] += np.expand_dims(self.s_reverse[1:(num_samples - 1)], axis=1)
            # above is equivalent to following but without the loop
            # for t in range(1, num_samples - 1):
            #     # broadcasting: (K, K) + (K, ) -> (K, K) + (1, K) row vector self.s_reverse[t] replicated K times
            #     ps[t] += self.s_reverse[t]

        # helper for partial derivative
        q = np.empty((num_samples, self.dim_k, self.dim_k), dtype=self._dtype)

        q[0].fill(0.0)
        softmax_1d_opt(ps[0, self.prev_label], out=q[0, self.prev_label])

        pst = np.reshape(ps[1:num_samples], (num_samples-1, self.dim_k * self.dim_k))
        q2 = np.reshape(q, (num_samples, self.dim_k * self.dim_k))
        softmax_2d_opt(pst, out=q2[1:num_samples])

        # element-wise addition of the N arrays of size (K, K)
        np.sum(q, axis=0, out=self.d_a_trans)

        self.d_a_trans[self.prev_label, self.labels[0]] -= 1
        # because the same indices can be repeated, this loop can't be vectorized using indexing
        for t in range(1, num_samples):
            self.d_a_trans[self.labels[t-1], self.labels[t]] -= 1

        # self.post_back_propagation_assert()

        return self.delta_err

    def post_backwards_assert(self):
        tolerance = 1e-6 if self._dtype == np.float32 else 1e-15

        num_samples = self.data.shape[0]

        all_seq_log_sum_exp = self.all_seq_log_sum_exp()

        tmp = np.log(np.sum(np.exp(self.ps[0][self.prev_label])))
        assert np.isclose(tmp, all_seq_log_sum_exp, rtol=tolerance, atol=tolerance)

        for t in range(1, num_samples):
            tmp = np.log(np.sum(np.exp(self.ps[t])))
            assert np.isclose(tmp, all_seq_log_sum_exp, rtol=tolerance, atol=tolerance)

    def forward_backwards(self, data, labels):
        loss = self.forward(data, labels)
        self.backwards()
        return loss, self._grad, self.delta_err

    def get_built_model(self):
        return self.a_trans.flatten()

    def get_built_gradient(self):
        return self.d_a_trans.flatten()

    def get_most_probable_seq(self, class_dtype=np.int32):
        """Decode the highest scoring sequence of tags using the Viterbi decoder.
        See the recurrence equations in https://en.wikipedia.org/wiki/Viterbi_algorithm for the more general case of
        HMM. The simplification here is that the visible and the hidden states are the same thing.
        Args:
            class_dtype: integer type of labels
        Returns:
            decoded_sequence: numpy array of shape (N, )
            decode_sequence_score: score of returned sequence, float
        """
        num_samples = self.data.shape[0]
        # Note: We can't use the trellis self.s we built in the forward method because self.s[t, j] is the log sum exp
        # of the scores of all possible (forward) paths ending at [t, j] while here we want the score of only one
        # forward path ending at [t, j]. So we build that simpler trellis here.
        # trellis[t, j] is the score of the most probable sequence that ends in label j at time t
        trellis = np.zeros_like(self.data)  # (T, K)
        # backpointers[t, j] is the previous label (at time t-1) in the path for the maximum probability sequence when
        # the label at time t is j
        backpointers = np.empty(self.data.shape, dtype=class_dtype)
        trellis[0] = self.a_trans[self.prev_label] + self.data[0]
        backpointers[0] = self.prev_label

        for t in range(1, num_samples):
            # (K, 1) + (K, K) -> (K, K)
            v = np.reshape(trellis[t - 1], (self.dim_k, 1)) + self.a_trans
            trellis[t] = self.data[t] + np.max(v, axis=0)
            backpointers[t] = np.argmax(v, axis=0)

        # assemble the highest probability path
        viterbi = np.empty(num_samples, dtype=class_dtype)
        viterbi[num_samples - 1] = np.argmax(trellis[num_samples - 1])
        for t in range(num_samples - 1, 0, -1):
            viterbi[t-1] = backpointers[t, viterbi[t]]

        viterbi_score = np.max(trellis[-1])

        return viterbi, viterbi_score
