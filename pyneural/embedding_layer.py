import numpy as np

import pyneural.embedding_cy as embedding_cy
from pyneural.neural_base import ComponentNN, BatchSequencesComponentNN, glorot_init, validate_x_and_lengths


class EmbeddingLayer(ComponentNN):
    def __init__(self, dim_k: int, dim_d: int, dtype, asserts_on=True):
        self.dim_k = dim_k
        self.dim_d = dim_d
        super(EmbeddingLayer, self).__init__(self.dim_k * self.dim_d, dtype)
        self.dk_threshold = int(0.40 * self.dim_k * self.dim_d)
        self._num_samples = 0
        self.embedding_matrix = None
        self._grad2 = None
        self.x = None
        self.asserts_on = asserts_on
        self.x_prev = None

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({'dim_k': self.dim_k, 'dim_d': self.dim_d})
        return d

    def model_copy_from_embedding_matrix(self, embedding_matrix_in):
        assert embedding_matrix_in.shape == (self.dim_k, self.dim_d)
        np.copyto(self._model, np.reshape(embedding_matrix_in, (self._num_p,)))

    def model_normal_init(self, sd: float):
        assert self._model is not None
        np.copyto(self.embedding_matrix, sd * np.random.standard_normal((self.dim_k, self.dim_d)).astype(self._dtype))

    def model_glorot_init(self):
        assert self._model is not None
        np.copyto(self.embedding_matrix, glorot_init((self.dim_k, self.dim_d)).astype(self._dtype))

    def forward(self, x):
        if self.asserts_on:
            assert x.ndim == 1
            assert np.issubdtype(x.dtype, np.integer)
            if x.shape[0] != 0:
                assert 0 <= x.min() and x.max() < self.dim_k

        self._num_samples = x.shape[0]
        self.x = x
        self.y = np.empty((x.shape[0], self.dim_d), dtype=self._dtype)
        embedding_cy.forward(self.x, self.embedding_matrix, self.y)

        return self.y

    def backwards(self, delta_err) -> None:
        if self.asserts_on:
            assert delta_err.shape == (self._num_samples, self.dim_d)
            assert delta_err.dtype == self._dtype

        if self.x_prev is None:
            # for large dimensionalities this sweeps potentially a large amount of memory, avoid
            self._grad2.fill(0.0)
        else:
            embedding_cy.backwards_zero_grad(self.x_prev, self._grad2)
        self.x_prev = np.copy(self.x) if self._num_samples < self.dk_threshold else None

        embedding_cy.backwards(self.x, delta_err, self._grad2)

        # derivative wr to non-continuous quantities is undefined
        return None

    def _set_model_references_in_place(self):
        self.embedding_matrix = np.reshape(self._model, (self.dim_k, self.dim_d))

    def _set_gradient_references_in_place(self):
        self._grad2 = np.reshape(self._grad, (self.dim_k, self.dim_d))

    def get_built_model(self):
        return np.reshape(self.embedding_matrix, (self._num_p,))

    def get_built_gradient(self):
        return np.reshape(self._grad2, (self._num_p,))


class EmbeddingLayerBatch(BatchSequencesComponentNN):
    def __init__(self, dim_k: int, dim_d: int, max_seq_length: int, max_batch_size: int, dtype, asserts_on=True):
        self.dim_k = dim_k
        self.dim_d = dim_d
        super(EmbeddingLayerBatch, self).__init__(self.dim_k * self.dim_d, max_seq_length, max_batch_size, dtype)
        self.y = np.empty((max_seq_length, max_batch_size, self.dim_d), dtype=dtype)
        self.dk_threshold = int(0.20 * self.dim_k * self.dim_d)
        self.embedding_matrix = None
        self._grad2 = None
        self.x = None
        self.asserts_on = asserts_on
        self._x_prev = None
        self._seq_lengths_prev = None
        self._prev_seq_length_dim_max = 0
        self._prev_num_sequences = 0

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({'dim_k': self.dim_k, 'dim_d': self.dim_d})
        return d

    def model_copy_from_embedding_matrix(self, embedding_matrix_in):
        assert embedding_matrix_in.shape == (self.dim_k, self.dim_d)
        np.copyto(self._model, np.reshape(embedding_matrix_in, (self._num_p, )))

    def model_normal_init(self, sd: float):
        assert self._model is not None
        np.copyto(self.embedding_matrix, sd * np.random.standard_normal((self.dim_k, self.dim_d)).astype(self._dtype))

    def model_glorot_init(self):
        assert self._model is not None
        np.copyto(self.embedding_matrix, glorot_init((self.dim_k, self.dim_d)).astype(self._dtype))

    def forward(self, x, seq_lengths):
        if self.asserts_on:
            validate_x_and_lengths(x, self._max_seq_length, self._max_num_sequences, seq_lengths)
            assert x.ndim == 2
            assert 0 <= x.min() and x.max() < self.dim_k  # requires 0-padded inputs!

        self._set_lengths(x, seq_lengths)

        if self.asserts_on:
            self._validate_zero_padding(x)

        self.x = x
        # reusing y instead of allocating each time is noticeably faster
        y = self.y[0:x.shape[0], 0:x.shape[1], :]
        embedding_cy.forward_batch(self.x, self.embedding_matrix, y, self._seq_lengths)
        return y

    def backwards(self, delta_err: np.array) -> None:
        if self.asserts_on:
            assert delta_err.shape == (self._curr_seq_length_dim_max, self._curr_num_sequences, self.dim_d)
            assert delta_err.dtype == self._dtype
            self._validate_zero_padding(delta_err)  # required by base class contract

        if self._prev_num_sequences == 0:
            # for large dimensionalities this sweeps potentially a large amount of memory, avoid
            self._grad2.fill(0.0)
        else:
            embedding_cy.backwards_batch_zero_grad(self._x_prev, self._seq_lengths_prev, self._grad2)

        # Copying is expensive, so the threshold for applying selective overwrites is lower than non-batched version.
        # a) allocating new memory each time and copying to it and b) re-using same memory and copy to it using slicing
        # seem to be about the same in run-time cost so we go for the simpler approach (new memory each time)
        if self._seq_lengths.shape[0] * self._curr_num_sequences < self.dk_threshold:
            self._x_prev = np.copy(self.x)
            self._seq_lengths_prev = np.copy(self._seq_lengths)
            self._prev_num_sequences = self._curr_num_sequences
        else:
            self._prev_num_sequences = 0

        embedding_cy.backwards_batch(self.x, delta_err, self._seq_lengths, self._grad2)

        # derivative wr to non-continuous quantities is undefined
        return None

    def _set_model_references_in_place(self):
        self.embedding_matrix = np.reshape(self._model, (self.dim_k, self.dim_d))

    def _set_gradient_references_in_place(self):
        self._grad2 = np.reshape(self._grad, (self.dim_k, self.dim_d))

    def get_built_model(self):
        return np.reshape(self.embedding_matrix, (self._num_p,))

    def get_built_gradient(self):
        return np.reshape(self._grad2, (self._num_p,))


class EmbeddingLayerPy(ComponentNN):
    """
    Correct implementation, but Cython-extended implementation is faster and therefore recommended.
    """
    def __init__(self, dim_k, dim_d, dtype, asserts_on=True):
        self.dim_k = dim_k
        self.dim_d = dim_d
        super(EmbeddingLayerPy, self).__init__(self.dim_k * self.dim_d, dtype)
        # dk_threshold is reasonable but mostly arbitrary, little attempt was made to determine it by measuring effect.
        # It was found clearly advantageous to selectively zero out gradient for _num_samples < < dim_k * dim_d as
        # opposed to setting the whole gradient array to 0.
        self.dk_threshold = int(0.05 * self.dim_k * self.dim_d) if self.dim_k * self.dim_d < 200000 else 10000
        self._num_samples = 0
        self.embedding_matrix = None
        self._grad2 = None
        self.x = None
        self.asserts_on = asserts_on
        self.x_prev = None

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({'dim_k': self.dim_k, 'dim_d': self.dim_d})
        return d

    def model_copy_from_embedding_matrix(self, embedding_matrix_in):
        assert embedding_matrix_in.shape == (self.dim_k, self.dim_d)
        np.copyto(self._model, np.reshape(embedding_matrix_in, (self._num_p,)))

    def model_normal_init(self, sd):
        assert self._model is not None
        np.copyto(self.embedding_matrix, sd * np.random.standard_normal((self.dim_k, self.dim_d)).astype(self._dtype))

    def model_glorot_init(self):
        assert self._model is not None
        np.copyto(self.embedding_matrix, glorot_init((self.dim_k, self.dim_d)).astype(self._dtype))

    def forward(self, x):
        if self.asserts_on:
            assert x.ndim == 1
            assert np.issubdtype(x.dtype, np.integer)
            if x.shape[0] != 0:
                assert 0 <= x.min() and x.max() < self.dim_k

        self._num_samples = x.shape[0]
        self.x = x
        self.y = self.embedding_matrix[x]
        return self.y

    def backwards(self, delta_err) -> None:
        if self.asserts_on:
            assert delta_err.shape == (self._num_samples, self.dim_d)
            assert delta_err.dtype == self._dtype

        if self.x_prev is None:
            # for large dimensionalities this sweeps potentially a large amount of memory, avoid
            self._grad2.fill(0.0)
        else:
            for i in self.x_prev:
                self._grad2[i, :].fill(0.0)
        self.x_prev = np.copy(self.x) if self._num_samples < self.dk_threshold else None

        # Unfortunately this can't be vectorized, because the same element self._grad2[i1] may need to be updated
        # multiple times. Numpy doc: "In particular, repeated indices do not result in the value getting added twice"
        for i in range(self._num_samples):
            self._grad2[self.x[i], :] += delta_err[i]  # += operator is in-place

        # derivative wr to non-continuous quantities is undefined
        return None

    def _set_model_references_in_place(self):
        self.embedding_matrix = np.reshape(self._model, (self.dim_k, self.dim_d))

    def _set_gradient_references_in_place(self):
        self._grad2 = np.reshape(self._grad, (self.dim_k, self.dim_d))

    def get_built_model(self):
        return np.reshape(self.embedding_matrix, (self._num_p,))

    def get_built_gradient(self):
        return np.reshape(self._grad2, (self._num_p,))


class EmbeddingLayerBatchPy(BatchSequencesComponentNN):
    """
    Correct implementation, but Cython-extended implementation is faster and therefore recommended.
    """
    def __init__(self, dim_k, dim_d, max_seq_length, max_batch_size, dtype, asserts_on=True):
        self.dim_k = dim_k
        self.dim_d = dim_d
        super(EmbeddingLayerBatchPy, self).__init__(self.dim_k * self.dim_d, max_seq_length, max_batch_size, dtype)
        self.embedding_matrix = None
        self._grad2 = None
        self.x = self.y = None
        self.asserts_on = asserts_on

    def get_display_dict(self):
        d = self._init_display_dict()
        d.update({'dim_k': self.dim_k, 'dim_d': self.dim_d})
        return d

    def model_copy_from_embedding_matrix(self, embedding_matrix_in):
        assert embedding_matrix_in.shape == (self.dim_k, self.dim_d)
        np.copyto(self._model, np.reshape(embedding_matrix_in, (self._num_p, )))

    def model_normal_init(self, sd):
        assert self._model is not None
        np.copyto(self.embedding_matrix, sd * np.random.standard_normal((self.dim_k, self.dim_d)).astype(self._dtype))

    def model_glorot_init(self):
        assert self._model is not None
        np.copyto(self.embedding_matrix, glorot_init((self.dim_k, self.dim_d)).astype(self._dtype))

    def forward(self, x, seq_lengths):
        if self.asserts_on:
            validate_x_and_lengths(x, self._max_seq_length, self._max_num_sequences, seq_lengths)
            assert x.ndim == 2
            assert 0 <= x.min() and x.max() < self.dim_k  # requires 0-padded inputs!

        self._set_lengths(x, seq_lengths)

        if self.asserts_on:
            self._validate_zero_padding(x)

        self.x = x

        # this does a lot of extra work for batches with short sequences, but much faster than alternative, see below
        self.y = self.embedding_matrix[x]
        # required by base class contract
        self._zero_pad_overwrite(self.y)  # this is (by far) the most expensive operation

        # following is much slower than the looking all indices and then zeroing out the extra ones
        # self.y = np.empty((x.shape[0], x.shape[1], self.dim_d), dtype=self._dtype)
        # for j in range(x.shape[1]):
        #     if 0 < seq_lengths[j]:
        #         # slicing twice also across non-leading dimensions, can't be fast
        #         self.y[:seq_lengths[j], j] = self.embedding_matrix[x[:seq_lengths[j], j]]
        #     if seq_lengths[j] < x.shape[0]:
        #         self.y[seq_lengths[j]:, j].fill(0.0)

        return self.y

    def backwards(self, delta_err) -> None:
        if self.asserts_on:
            assert delta_err.shape == (self._curr_seq_length_dim_max, self._curr_num_sequences, self.dim_d)
            assert delta_err.dtype == self._dtype
            self._validate_zero_padding(delta_err)  # required by base class contract

        # we can't use here the dk_threshold trick as in EmbeddingLayerPy because that involves 2 loops which is
        # prohibitively slow
        self._grad2.fill(0.0)

        # Unfortunately this can't be vectorized, because the same element self._grad2[i1] may need to be updated
        # multiple times. Numpy doc: "In particular, repeated indices do not result in the value getting added twice"
        # Memory access pattern of double loop is not ideal, but restructuring the loops as commented out did not
        # improve run-time even for large dimensionalities.
        for j in range(self._curr_num_sequences):
            for i in range(self._seq_lengths[j]):
                self._grad2[self.x[i, j], :] += delta_err[i, j]  # += operator is in-place

        # min_seq_length = np.min(self._seq_lengths)
        # for i in range(min_seq_length):
        #     for j in range(self._curr_num_sequences):
        #         self._grad2[self.x[i, j], :] += delta_err[i, j]
        # for j in range(self._curr_num_sequences):
        #     for i in range(min_seq_length, self._seq_lengths[j]):
        #         self._grad2[self.x[i, j], :] += delta_err[i, j]

        # derivative wr to non-continuous quantities is undefined
        return None

    def _set_model_references_in_place(self):
        self.embedding_matrix = np.reshape(self._model, (self.dim_k, self.dim_d))

    def _set_gradient_references_in_place(self):
        self._grad2 = np.reshape(self._grad, (self.dim_k, self.dim_d))

    def get_built_model(self):
        return np.reshape(self.embedding_matrix, (self._num_p,))

    def get_built_gradient(self):
        return np.reshape(self._grad2, (self._num_p,))
