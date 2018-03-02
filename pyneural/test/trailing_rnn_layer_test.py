import unittest

import numpy as np

import gradient_check_test_shared as gcs
import pyneural.trailing_rnn_layer as trl
import rnn_batch_layer_test as rblt
from pyneural.ce_l2_loss import LayerWithL2Loss, BatchSequencesWithL2Loss


def _create_random_data_non_full_batch(rnn_batch):
    assert isinstance(rnn_batch, trl.TrailingRnnBatchLayer)

    x, _, seq_lengths = rblt.create_random_data_non_full_batch(rnn_batch)

    dtype = rnn_batch.get_dtype()
    dim_d, dim_h = rnn_batch.get_dimensions()
    y_true = np.random.standard_normal((rnn_batch.get_max_batch_size(), dim_h)).astype(dtype)
    for i in xrange(rnn_batch.get_max_batch_size()):
        if seq_lengths[i] == 0:
            y_true[i].fill(0.0)

    return x, y_true, seq_lengths


def _create_random_data_full_batch(rnn_batch):
    assert isinstance(rnn_batch, trl.TrailingRnnBatchLayer)

    x, _, seq_lengths = rblt.create_random_data_full_batch(rnn_batch)

    dtype = rnn_batch.get_dtype()
    dim_d, dim_h = rnn_batch.get_dimensions()
    y_true = np.random.standard_normal((rnn_batch.get_max_batch_size(), dim_h)).astype(dtype)

    return x, y_true, seq_lengths


class TrailingRnnTest(gcs.GradientCheckTestShared):

    def test_gradients(self):
        num_samples = 20
        dim_x, dim_h = 5, 3

        dtype, tolerance = (np.float64, 1e-8)

        tr_layer = trl.TrailingRnnLayer(dim_x, dim_h, num_samples, dtype, activation="tanh")
        loss_and_layer = LayerWithL2Loss(tr_layer)

        np.random.seed(seed=47)
        model = 0.1 * np.random.standard_normal(loss_and_layer.get_num_p()).astype(dtype)
        x = np.random.standard_normal((num_samples, dim_x)).astype(dtype)
        y = np.random.standard_normal(dim_h).astype(dtype)
        h_init = 0.001 * np.random.standard_normal(dim_h).astype(dtype)

        loss_and_layer.init_parameters_storage(model=model)

        self.do_param_gradient_check(loss_and_layer, x, y, tolerance, h_init)
        self.do_input_gradient_check(loss_and_layer, x, y, tolerance, h_init)

        tr_layer2 = trl.TrailingRnnLayer2(dim_x, dim_h, num_samples, dtype=dtype, activation="tanh")
        loss_and_layer = LayerWithL2Loss(tr_layer2)
        loss_and_layer.init_parameters_storage(model=model)

        self.do_param_gradient_check(loss_and_layer, x, y, tolerance, h_init)
        self.do_input_gradient_check(loss_and_layer, x, y, tolerance, h_init)

    def test_gradients_batched(self):
        dim_x, dim_h = 5, 3
        batch_size = 4
        max_seq_length = 10
        dtype, tolerance = np.float64, 1e-8

        tr_layer = trl.TrailingRnnBatchLayer(dim_x, dim_h, max_seq_length, batch_size, dtype=dtype, activation="tanh")
        loss_and_layer = BatchSequencesWithL2Loss(tr_layer)

        np.random.seed(seed=47)
        model = 0.1 * np.random.standard_normal((tr_layer.get_num_p(),)).astype(dtype)
        h_init = 0.01 * np.random.standard_normal((batch_size, dim_h)).astype(dtype)

        loss_and_layer.init_parameters_storage(model=model)

        tr_layer_gru = trl.TrailingGruBatchLayer(dim_x, dim_h, max_seq_length, batch_size, dtype)
        loss_and_layer_gru = BatchSequencesWithL2Loss(tr_layer_gru)
        model = 0.1 * np.random.standard_normal((tr_layer_gru.get_num_p(),)).astype(dtype)
        loss_and_layer_gru.init_parameters_storage(model=model)

        x, y, seq_lengths = _create_random_data_non_full_batch(tr_layer)
        self.do_param_batched_gradient_check(loss_and_layer, x, y, seq_lengths, tolerance, h_init)
        self.do_param_batched_gradient_check(loss_and_layer_gru, x, y, seq_lengths, tolerance, h_init)

        # input gradient check possible only for full batch of data
        x, y, seq_lengths = _create_random_data_full_batch(tr_layer)
        self.do_input_batched_gradient_check(loss_and_layer, x, y, seq_lengths, tolerance, h_init)
        self.do_param_batched_gradient_check(loss_and_layer, x, y, seq_lengths, tolerance, h_init)

        self.do_input_batched_gradient_check(loss_and_layer_gru, x, y, seq_lengths, tolerance, h_init)
        self.do_param_batched_gradient_check(loss_and_layer_gru, x, y, seq_lengths, tolerance, h_init)

    def test_zero_length_sequence_batched(self):
        dim_x, dim_h = 5, 3
        batch_size = 4
        max_seq_length = 8
        dtype, tolerance = np.float64, 1e-8

        tr_layer = trl.TrailingRnnBatchLayer(dim_x, dim_h, max_seq_length, batch_size, dtype, "sigmoid")
        loss_and_layer = BatchSequencesWithL2Loss(tr_layer)

        np.random.seed(seed=47)
        model = 0.1 * np.random.standard_normal((tr_layer.get_num_p(),)).astype(dtype)
        h_init = 0.01 * np.random.standard_normal((batch_size, dim_h)).astype(dtype)

        loss_and_layer.init_parameters_storage(model)
        tr_layer.set_init_h(h_init)

        x, y, seq_lengths = _create_random_data_non_full_batch(tr_layer)

        # set second sequence to have 0 length (first was created already with 0 length)
        seq_length_sav = seq_lengths[1]
        seq_lengths[1] = 0

        # having the corresponding x to be all 0s is required and produces no error
        x[0:seq_length_sav, 1, :] = 0.0
        loss_and_layer.forward_backwards(x, y, seq_lengths)

        # setting the corresponding x and y to be all 0s is accepted
        y[1, :] = 0.0
        _, _, delta_err = loss_and_layer.forward_backwards(x, y, seq_lengths)
        self.assertTrue(np.alltrue(np.equal(delta_err[:, 0, :], 0.0)))


if __name__ == "__main__":
    unittest.main()
