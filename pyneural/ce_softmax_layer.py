import numpy as np

from softmax import softmax_1d_opt, softmax_2d_opt, softmax_3d_opt
from neural_base import (
    LossNN, BatchSequencesLossNN, glorot_init,
    validate_x_and_lengths, validate_zero_padding, zero_pad_overwrite
)


class CESoftmaxLayer(LossNN):
    """
    Scalar softmax top-layer and connections (synapses) (projection layer) with the lower layer
    """

    def __init__(self, dim_k, dim_d, dtype, asserts_on=True):
        self.dim_k, self.dim_d = dim_k, dim_d
        num_params = self.dim_k * self.dim_d + self.dim_k
        super(CESoftmaxLayer, self).__init__(num_params, dtype)
        self.p_hat = None
        self.delta_err = None
        self.w = self.b = None
        self.dw = self.db = None
        self.asserts_on = asserts_on

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({"dim_k": self.dim_k, "dim_d": self.dim_d})
        return d

    def model_normal_init(self, sd):
        np.copyto(self.w, sd * np.random.standard_normal((self.dim_k, self.dim_d)).astype(self._dtype))
        self.b.fill(0.0)

    def model_glorot_init(self):
        np.copyto(self.w, glorot_init((self.dim_k, self.dim_d)).astype(self._dtype))
        self.b.fill(0.0)

    def model_glorot_init_with_bias(self, bias):
        np.copyto(self.w, glorot_init((self.dim_k, self.dim_d)).astype(self._dtype))
        np.copyto(self.b, bias)

    def forward_single(self, x, label):
        if self.asserts_on:
            assert x.shape == (self.dim_d,)
            assert x.dtype == self._dtype

        self._x, self._y_true = x, label

        # (K, D) x (D, ) + (K, ) -> (K, )
        y = np.dot(self.w, self._x) + self.b
        self.p_hat = softmax_1d_opt(y)

        return - np.log(self.p_hat[self._y_true])

    def backwards_single(self):
        ds = np.copy(self.p_hat)
        ds[self._y_true] -= 1

        np.copyto(self.db, ds)
        self.dw = np.outer(ds, self._x, out=self.dw)
        self.delta_err = np.dot(ds, self.w)  # (K, ) x (K, D) -> (D, )

        return self.delta_err

    def forward_backwards_single(self, x, label):
        loss = self.forward_single(x, label)
        self.backwards_single()
        return loss, self.get_gradient(), self.delta_err

    def forward(self, x, labels):
        if self.asserts_on:
            assert x.ndim == 2 and x.shape[1] == self.dim_d
            assert x.dtype == self._dtype
            assert labels.shape == (x.shape[0],)
            assert np.issubdtype(labels.dtype, np.integer)
            assert labels.max() < self.dim_k

        self._x, self._y_true = x, labels

        num_samples = x.shape[0]

        # ((K, D) x (D, N))^T = (N, D) x (D, K) = (N, K)
        # broadcasting: (N, K) + (K, ) = (N, K) + (1, K) -> (N, K) + (N, K)
        y = np.dot(self._x, self.w.T) + self.b
        self.p_hat = softmax_2d_opt(y)

        loss = - np.log(self.p_hat[xrange(num_samples), self._y_true])

        return np.sum(loss)

    def get_most_probable(self):
        # CAUTION: in back propagation self.p_hat is modified in-place. This method returns correct results only after
        # the forward pass and before the back propagation pass. The assertion enforces that.
        assert np.amin(self.p_hat) >= 0  # make sure no back propagation applied
        return np.argmax(self.p_hat, axis=1)

    def backwards(self):
        num_samples = self._x.shape[0]

        # self.p_hat will be modified in-place to be equal to softmax derivative w.r. to softmax inputs
        # for readability name it delta_s
        delta_s = self.p_hat  # not a copy, p_hat modified in place
        # numpy fancy indexing required below, slicing first dim as 0:num_samples is wrong
        delta_s[xrange(num_samples), self._y_true] -= 1

        np.sum(delta_s, axis=0, out=self.db)

        # (K, N) x (N, D) is sum over the N samples of (K, 1) x (1, D)
        np.dot(delta_s.T, self._x, out=self.dw)

        self.delta_err = np.dot(delta_s, self.w)  # (N, K) x (K, D) -> (N, D)

        return self.delta_err

    def forward_backwards(self, x, labels):
        self._x, self._y_true = x, labels

        loss = self.forward(x, labels)
        self.backwards()
        return loss, self._grad, self.delta_err

    def __unpack_model_or_grad(self, params):
        kxh = self.dim_k * self.dim_d
        w_hy = np.reshape(params[0:kxh], (self.dim_k, self.dim_d))
        b_y = np.reshape(params[kxh:(kxh + self.dim_k)], (self.dim_k, ))
        return w_hy, b_y

    def _set_model_references_in_place(self):
        self.w, self.b = self.__unpack_model_or_grad(self._model)

    def _set_gradient_references_in_place(self):
        self.dw, self.db = self.__unpack_model_or_grad(self._grad)

    def slice_model(self, classes_to_keep):
        w1 = self.w[classes_to_keep, :]
        b1 = self.b[classes_to_keep]
        return np.concatenate((w1.flatten(), b1))

    def get_built_model(self):
        return np.concatenate((self.w.flatten(), self.b))

    def get_built_gradient(self):
        return np.concatenate((self.dw.flatten(), self.db))


class CESoftmaxLayerFixedLength(LossNN):
    """ Optimized version of CESoftmaxLayer when the input length is known and fixed.
    
    But it is only marginally faster and only for large model dimensions and input sizes.

    Do not use! Simply not better enough.

    Apparently the overhead of additional python function calls outweights the cost of system calls for memory
    allocation, or the bottleneck is elsewhere. It turns out that the cost of fancy indexing in p_hat -= 1
    is about as much as the cost of (K, N) x (N, D) outer product.
    """

    def __init__(self, dim_k, dim_d, seq_length, dtype, asserts_on=True):
        self.dim_k, self.dim_d, self.seq_length = dim_k, dim_d, seq_length
        num_params = self.dim_k * self.dim_d + self.dim_k
        super(CESoftmaxLayerFixedLength, self).__init__(num_params, dtype)
        self.x, self.labels = None, None
        self.p_hat = np.empty((self.seq_length, self.dim_k), dtype=dtype)
        self.delta_err = np.empty((self.seq_length, self.dim_d), dtype=dtype)
        self.w = self.b = None
        self.dw = self.db = None
        self.asserts_on = asserts_on

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({"dim_k": self.dim_k, "dim_d": self.dim_d, 'seq_length': self.seq_length})
        return d

    def model_normal_init(self, sd):
        np.copyto(self.w, sd * np.random.standard_normal((self.dim_k, self.dim_d)).astype(self._dtype))
        self.b.fill(0.0)

    def model_glorot_init(self):
        np.copyto(self.w, glorot_init((self.dim_k, self.dim_d)).astype(self._dtype))
        self.b.fill(0.0)

    def model_glorot_init_with_bias(self, bias):
        np.copyto(self.w, glorot_init((self.dim_k, self.dim_d)).astype(self._dtype))
        np.copyto(self.b, bias)

    def forward(self, x, labels):
        if self.asserts_on:
            assert x.shape == (self.seq_length, self.dim_d)
            assert x.dtype == self._dtype
            assert labels.shape == (self.seq_length, )
            assert np.max(labels) < self.dim_k

        self.x = x
        self.labels = labels

        # ((K, D) x (D, N))^T = (N, D) x (D, K) = (N, K)
        # broadcasting: (N, K) + (K, ) = (N, K) + (1, K) -> (N, K) + (N, K)
        # y = np.dot(self.x, self.w.T) + self.b
        y = self.p_hat
        np.dot(self.x, self.w.T, out=y)
        np.add(y, self.b, out=y)
        softmax_2d_opt(y, out=y)

        loss = - np.log(self.p_hat[xrange(self.seq_length), self.labels])

        return np.sum(loss)

    def get_most_probable(self):
        # CAUTION: in back propagation self.p_hat is modified in-place. This method returns correct results only after
        # the forward pass and before the back propagation pass. The assertion enforces that.
        assert np.amin(self.p_hat) >= 0  # make sure no back propagation applied
        return np.argmax(self.p_hat, axis=1)

    def backwards(self):
        # self.p_hat will be modified in-place to be equal to softmax derivative w.r. to softmax inputs
        # for readability name it delta_s
        delta_s = self.p_hat  # not a copy, p_hat modified in place
        # numpy fancy indexing required below, slicing first dim as 0:num_samples is wrong
        delta_s[xrange(self.seq_length), self.labels] -= 1

        np.sum(delta_s, axis=0, out=self.db)

        # (K, N) x (N, D) is sum over the N samples of (K, 1) x (1, D)
        np.dot(delta_s.T, self.x, out=self.dw)

        # (N, K) x (K, D) -> (N, D)
        np.dot(delta_s, self.w, out=self.delta_err)

        return self.delta_err

    def forward_backwards(self, x, labels):
        loss = self.forward(x, labels)
        self.backwards()
        return loss, self._grad, self.delta_err

    def __unpack_model_or_grad(self, params):
        kxh = self.dim_k * self.dim_d
        w_hy = np.reshape(params[0:kxh], (self.dim_k, self.dim_d))
        b_y = np.reshape(params[kxh:(kxh + self.dim_k)], (self.dim_k,))
        return w_hy, b_y

    def _set_model_references_in_place(self):
        self.w, self.b = self.__unpack_model_or_grad(self._model)

    def _set_gradient_references_in_place(self):
        self.dw, self.db = self.__unpack_model_or_grad(self._grad)

    def get_built_model(self):
        return np.concatenate((self.w.flatten(), self.b))

    def get_built_gradient(self):
        return np.concatenate((self.dw.flatten(), self.db))


class CESoftmaxLayerBatch(BatchSequencesLossNN):
    """ Scalar softmax top-layer and connections (synapses) (projection layer) with the lower layer

    Not necessarily faster than a loop over CESoftmaxLayer, which is surprising and needs further investigation.
    Specifically, it scales very poorly with max_seq_length.

    Unfortunately duplicates a lot of BatchSequencesComponentNN functionality.

    In ce_softmax_layer_test.py it was found that it is slower(!) than CESoftmaxLayer
    for dim_k=500, max_seq_length=50, batch_size=20
    """

    def __init__(self, dim_k, dim_d, max_seq_length, batch_size, dtype, asserts_on=True):
        self.dim_k, self.dim_d = dim_k, dim_d
        self.p_hat = None
        self.delta_err = None
        num_params = self.dim_k * self.dim_d + self.dim_k
        super(CESoftmaxLayerBatch, self).__init__(num_params, dtype)
        self.w_hy = self.b = None
        self.dw_hy = self.db = None
        self.kxd_array = np.empty((self.dim_k, self.dim_d), dtype=self._dtype)
        self.asserts_on = asserts_on
        self._max_seq_length = max_seq_length  # maximum length of dimension 0 (time) of x for all batches
        self._max_num_sequences = batch_size  # maximum length of dimension 1 (sequence) of x for all batches
        if self._max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive integer, supplied %s" % str(max_seq_length))
        if self._max_num_sequences <= 0:
            raise ValueError("batch_size must be positive integer, supplied %s" % str(batch_size))
        # minimum and max lengths of valid values in dimension 0 (time) input x
        self._curr_min_seq_length = 0
        self._curr_max_seq_length = 0
        self._curr_seq_length_dim_max = 0  # current input x.shape[0]
        # It is allowed that no sequence in the batch has length equal to the maximum dimension, i.e. it is allowed
        # that _curr_max_seq_length < _curr_seq_length_dim_max
        self._curr_num_sequences = 0  # current input x.shape[1]
        self._seq_lengths = None  # 1-d np.array

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({"dim_k": self.dim_k, "dim_d": self.dim_d})
        return d

    def model_normal_init(self, sd):
        np.copyto(self.w_hy, sd * np.random.standard_normal((self.dim_k, self.dim_d)).astype(self._dtype))
        self.b.fill(0.0)

    def model_glorot_init(self):
        np.copyto(self.w_hy, glorot_init((self.dim_k, self.dim_d)).astype(self._dtype))
        self.b.fill(0.0)

    def get_max_batch_size(self):
        return self._max_num_sequences

    def get_max_seq_length(self):
        return self._max_seq_length

    def forward(self, x, labels, seq_lengths):
        if self.asserts_on:
            validate_x_and_lengths(x, x.shape[0], x.shape[1], seq_lengths)
            assert x.ndim == 3
            assert x.shape[2] == self.dim_d
            assert labels.shape == (x.shape[0], x.shape[1])
            assert np.issubdtype(labels.dtype, np.integer)

        self._curr_seq_length_dim_max = x.shape[0]
        self._curr_num_sequences = x.shape[1]

        self._x, self._y_true, self._seq_lengths = x, labels, seq_lengths

        if self.asserts_on:
            validate_zero_padding(x, self._curr_seq_length_dim_max, self._curr_num_sequences, seq_lengths)

        # There is no guarantee that elements in y_true after the sequence lengths are 0 and we MUST ignore them, so
        # zero out their contribution to error. It is allowed to change labels in-place.
        zero_pad_overwrite(labels, self._curr_seq_length_dim_max, self._curr_num_sequences, seq_lengths)

        # ((K, D) x (D, N))^T = (N, K)
        # without batching: (N, D) x (D, K) = (N, K)
        # broadcasting: (N, K) + (K, ) = (N, K) + (1, K) -> (N, K) + (N, K)
        # here with batching: (M, N, D) x (D, K) = (M, N, K)
        # broadcasting: (M, N, K) + (1, K) = (M, N, K) + (M, N, K)
        y_fwd = np.dot(self._x, self.w_hy.T) + self.b
        # y_fwd may contain invalid out-of-sequences elements that will be ignored

        curr_min_seq_length = self._curr_min_seq_length = self._seq_lengths.min()
        self._curr_max_seq_length = self._seq_lengths.max()

        # at back propagation, we rely on elements of the p_hat array outside of valid sequences to be 0 and
        # for its 1st dim to be equal to x.shape[0]
        self.p_hat = np.zeros((self._curr_seq_length_dim_max, self._curr_num_sequences, self.dim_k),
                              dtype=self._dtype)
        loss_v = np.zeros((self._curr_max_seq_length, self._curr_num_sequences), dtype=self._dtype)

        if curr_min_seq_length > 0:
            # optimization: here we vectorize the common lengths of sequences in the batch
            # testing showed that versions with softmax_3d_opt() and softmax_3d() have about the same running cost,
            # therefore something else has much higher cost
            softmax_3d_opt(y_fwd[0:curr_min_seq_length], out=self.p_hat[0:curr_min_seq_length])  # (M, N, K)
            # self.p_hat[0:curr_min_seq_length] = softmax_3d(y_fwd[0:curr_min_seq_length])  # (M, N, K)
            for i in xrange(curr_min_seq_length):
                p_hat_i = self.p_hat[i]
                loss_v[i] = - np.log(
                    p_hat_i[range(self._curr_num_sequences), self._y_true[i, 0:self._curr_num_sequences]])

        if curr_min_seq_length < self._curr_max_seq_length:
            for j in xrange(self._curr_num_sequences):
                seq_length = self._seq_lengths[j]
                if curr_min_seq_length < seq_length:
                    softmax_2d_opt(y_fwd[curr_min_seq_length:seq_length, j, :],
                                   out=self.p_hat[curr_min_seq_length:seq_length, j, :])
                    p_hat_j = self.p_hat[curr_min_seq_length:seq_length, j, :]   # (L, K)
                    loss_v[curr_min_seq_length:seq_length, j] = \
                        - np.log(p_hat_j[range(seq_length - curr_min_seq_length),
                                         self._y_true[curr_min_seq_length:seq_length, j]])
                # CAREFUL: p_hat_j[curr_min_seq_length:seq_length, ..]
                # is NOT same as p_hat_j[range(curr_min_seq_length, seq_length), .. ]  ("advanced/fancy indexing")
                # the version with range is the correct one

        return np.sum(loss_v)

    def backwards(self):
        curr_min_seq_length = self._curr_min_seq_length

        if curr_min_seq_length > 0:
            # optimization: here we vectorize the common lengths of sequences in the batch
            for i in xrange(curr_min_seq_length):
                p_hat_i = self.p_hat[i]
                p_hat_i[xrange(self._curr_num_sequences), self._y_true[i, 0:self._curr_num_sequences]] -= 1

        if curr_min_seq_length < self._curr_max_seq_length:
            for j in xrange(self._curr_num_sequences):
                seq_length = self._seq_lengths[j]
                if curr_min_seq_length < seq_length:
                    p_hat_j = self.p_hat[curr_min_seq_length:seq_length, j, :]  # (L, K)
                    p_hat_j[xrange(seq_length - curr_min_seq_length),
                            self._y_true[curr_min_seq_length:seq_length, j]] -= 1

        # the above modified self.p_hat in-place to be equal to softmax derivative w.r. to softmax inputs
        # for readability name it delta_s
        delta_s = self.p_hat

        # non-batch was: (N, K), now with batch: (M, N, K)
        # reduce_sum (M, N, K) to (K, )
        np.sum(delta_s, axis=(0, 1), out=self.db)

        # we can't easily sum over the M, N dimensions using matrix multiplications, so we use a loop for the first
        # dimension only (time)
        self.dw_hy.fill(0.0)
        kxd_array = self.kxd_array
        for t in xrange(self._curr_max_seq_length):
            # (K, D) = (K, N) x (N, D) is the sum of outer products (K, 1) x (1, D) over the N sequences at time t
            # self.dw_hy += np.dot(delta_s[t].T, self.x[t])
            np.dot(delta_s[t].T, self._x[t], out=kxd_array)
            self.dw_hy += kxd_array
        # (D, N) x (M, N, K) = (D, M, K), then np.sum(axis=1) ?

        # without batching:   (N, K) x (K, D) = (N, D)
        # here with batching: (M, N, K) x (K, D) = (M, N, D)
        self.delta_err = np.dot(delta_s, self.w_hy)
        return self.delta_err

    def forward_backwards(self, x, y_true, seq_lengths):
        loss = self.forward(x, y_true, seq_lengths)
        self.backwards()
        return loss, self.get_gradient(), self.delta_err

    def __unpack_model_or_grad(self, params):
        kxh = self.dim_k * self.dim_d
        w_hy = np.reshape(params[0:kxh], (self.dim_k, self.dim_d))
        b_y = np.reshape(params[kxh:(kxh + self.dim_k)], (self.dim_k, ))
        return w_hy, b_y

    def _set_model_references_in_place(self):
        self.w_hy, self.b = self.__unpack_model_or_grad(self._model)

    def _set_gradient_references_in_place(self):
        self.dw_hy, self.db = self.__unpack_model_or_grad(self._grad)

    def get_built_model(self):
        return np.concatenate((self.w_hy.flatten(), self.b))

    def get_built_gradient(self):
        return np.concatenate((self.dw_hy.flatten(), self.db))
