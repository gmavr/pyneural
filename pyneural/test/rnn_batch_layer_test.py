import unittest

import numpy as np

import pyneural.rnn_batch_layer as rbl
import pyneural.rnn_layer as rl
import pyneural.test.gradient_check_test_shared as gcs
from pyneural.ce_l2_loss import BatchSequencesWithL2Loss
from pyneural.neural_base import BatchSequencesComponentNN


def _create_random_params(rnn_batch):
    assert isinstance(rnn_batch, rbl.RnnBatchLayer)

    batch_size = rnn_batch.get_max_batch_size()
    dtype = rnn_batch.get_dtype()
    dim_d, dim_h = rnn_batch.get_dimensions()

    params = 0.1 * np.random.standard_normal((rnn_batch.get_num_p(),)).astype(dtype)
    h_init = 0.01 * np.random.standard_normal((batch_size, dim_h)).astype(dtype)

    return params, h_init


def create_random_data_non_full_batch(rnn_batch):
    assert isinstance(rnn_batch, BatchSequencesComponentNN)

    batch_size = rnn_batch.get_max_batch_size()
    max_seq_length = rnn_batch.get_max_seq_length()
    dtype = rnn_batch.get_dtype()
    dim_d, dim_h = rnn_batch.get_dimensions()

    assert max_seq_length > 3  # we subtract 3 and want the result strictly greater than 0

    # set first sequence to 0 length to test that case
    # set all other sequences to random lengths greater than 0 but smaller than the maximum allowed length that the
    # layer was initialized with
    seq_lengths = np.random.randint(max_seq_length - 3, max_seq_length, batch_size)
    seq_lengths[0] = 0

    x = np.zeros((max_seq_length, batch_size, dim_d), dtype=dtype)
    y_true = np.zeros((max_seq_length, batch_size, dim_h), dtype=dtype)
    for i in range(1, batch_size):
        seq_length = seq_lengths[i]
        x[0:seq_length, i, :] = np.random.standard_normal((seq_length, dim_d)).astype(dtype)
        y_true[0:seq_length, i, :] = np.random.standard_normal((seq_length, dim_h)).astype(dtype)

    return x, y_true, seq_lengths


def create_random_data_full_batch(rnn_batch):
    assert isinstance(rnn_batch, BatchSequencesComponentNN)

    batch_size = rnn_batch.get_max_batch_size()
    max_seq_length = rnn_batch.get_max_seq_length()
    dtype = rnn_batch.get_dtype()
    dim_d, dim_h = rnn_batch.get_dimensions()

    seq_lengths = np.empty(batch_size, dtype=int)
    seq_lengths.fill(max_seq_length)

    x = np.random.standard_normal((max_seq_length, batch_size, dim_d)).astype(dtype)
    y_true = np.random.standard_normal((max_seq_length, batch_size, dim_h)).astype(dtype)

    return x, y_true, seq_lengths


class TestRnnLayer(gcs.GradientCheckTestShared):

    def test_gradients_batched(self):
        dim_x, dim_h = 8, 3
        batch_size = 5
        max_seq_length = 10
        dtype, tolerance = np.float64, 5e-8

        batch_layer = rbl.RnnBatchLayer(dim_x, dim_h, max_seq_length, batch_size, dtype=dtype, activation="tanh")
        loss_and_layer = BatchSequencesWithL2Loss(batch_layer)

        np.random.seed(47)
        params, h_init = _create_random_params(batch_layer)

        x, y, seq_lengths = create_random_data_non_full_batch(batch_layer)
        # make sure that we are testing the case of a 0 length sequence
        assert seq_lengths[0] == 0

        # shrink data set to contain fewer sequences than the batch size set at initialization
        x = x[:, :-1]
        y = y[:, :-1]
        seq_lengths = seq_lengths[:-1]

        loss_and_layer.init_parameters_storage(params)

        self.do_param_batched_gradient_check(loss_and_layer, x, y, seq_lengths, tolerance, h_init)

        # input gradient check possible only for full batch of data
        x, y, seq_lengths = create_random_data_full_batch(batch_layer)
        self.do_input_batched_gradient_check(loss_and_layer, x, y, seq_lengths, tolerance, h_init)
        self.do_param_batched_gradient_check(loss_and_layer, x, y, seq_lengths, tolerance, h_init)

    def test_zero_length_sequence_batched(self):
        dim_x, dim_h = 3, 2
        batch_size = 5
        max_seq_length = 8
        dtype = np.float64

        batch_layer = rbl.RnnBatchLayer(dim_x, dim_h, max_seq_length, batch_size, dtype=dtype, activation="tanh")
        loss_and_layer = BatchSequencesWithL2Loss(batch_layer)

        np.random.seed(47)
        params, h_init = _create_random_params(batch_layer)

        loss_and_layer.init_parameters_storage(params)
        batch_layer.set_init_h(h_init)

        x, y, seq_lengths = create_random_data_non_full_batch(batch_layer)

        # set second sequence to have 0 length (first was created already with 0 length)
        seq_length_sav = seq_lengths[1]
        seq_lengths[1] = 0

        # shrink data set to contain fewer sequences than the batch size set at initialization
        x = x[:, :-1]
        y = y[:, :-1]
        seq_lengths = seq_lengths[:-1]

        # The second sequence has 0 length but its contents are not zero.
        # Verify that this is detected when asserts_on == True
        with self.assertRaises(AssertionError):
            loss_and_layer.forward_backwards(x, y, seq_lengths)

        # Setting the corresponding x to be all 0s is now accepted.
        # Verify that y beyond the sequence end is accepted and ignored.
        x[0:seq_length_sav, 1, :] = 0.0
        _, _, delta_err = loss_and_layer.forward_backwards(x, y, seq_lengths)
        self.assertTrue(np.all(np.equal(delta_err[:, 0, :], 0.0)))

    def test_batching_equivalence(self):
        """Verifies that a batched invocation of N sequences and N non-batched invocations of 1 sequence return the same
         results for the forward and backwards passes.
        """
        dim_d, dim_h = 3, 2
        batch_size = 5
        max_batch_size = batch_size + 1
        max_seq_length = 4
        bptt_steps = max_seq_length
        dtype, tolerance = np.float64, 1e-14

        # intentionally all sequences shorter than max_seq_length
        data_t = np.array([
            [[3.1, 0.1, -7.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.1, 2.1, -3.2], [2.1, -4.5, 3.4], [5.0, -3.1, -0.6], [0.0, 0.0, 0.0]],
            [[1.5, -0.7, 5.1], [-9.1, 2.7, 0.6], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[-5.0, 0.1, 0.2], [2.7, 9.1, -2.0], [2.1, -1.5, 1.4], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ], dtype=dtype)
        seq_lengths = np.array([1, 3, 2, 3, 0], dtype=int)

        self.assertEqual(data_t.shape, (batch_size, max_seq_length, dim_d))
        self.assertEqual(seq_lengths.shape, (batch_size, ))

        data = np.transpose(data_t, (1, 0, 2))

        delta_upper_t = np.array([
            [[-1.1, 2.3], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[-0.1, -6.1], [1.5, 0.1], [2.1, -5.1], [0.0, 0.0]],
            [[0.8, 2.1], [-2.1, 4.5], [0.0, 0.0], [0.0, 0.0]],
            [[3.2, -5.4], [3.2, 2.1], [-4.6, -1.2], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ], dtype=dtype)

        self.assertEqual(delta_upper_t.shape, (batch_size, max_seq_length, dim_h))

        delta_upper = np.transpose(delta_upper_t, (1, 0, 2))

        # forward propagation

        rnn_layer = rl.RnnLayer(dim_d, dim_h, max_seq_length, dtype, "tanh", bptt_steps)
        rnn_batch_layer_1 = rbl.RnnBatchLayer(dim_d, dim_h, max_seq_length, max_batch_size, dtype,
                                              bptt_steps=bptt_steps)
        rnn_batch_layer_2 = rbl.RnnBatchLayer(dim_d, dim_h, max_seq_length, 1, dtype, bptt_steps=bptt_steps)
        rnn_batch_layer_3 = rbl.RnnBatchLayerTime2nd(dim_d, dim_h, max_seq_length, max_batch_size, dtype,
                                                     bptt_steps=bptt_steps)

        model = 0.1 * np.random.standard_normal((rnn_layer.get_num_p(),)).astype(dtype)
        hs_init = 0.01 * np.random.standard_normal((batch_size, dim_h)).astype(dtype)

        # we do not need to allocate our own gradient buffer for testing back propagation, so this is an additional
        # higher level test
        grad_storage_1 = np.empty((rnn_batch_layer_1.get_num_p(),), dtype=dtype)
        rnn_batch_layer_1.set_gradient_storage(grad_storage_1)

        # all following objects share the same model memory, but each has its own separate gradient memory
        rnn_layer.init_parameters_storage(model)
        rnn_batch_layer_1.init_parameters_storage(model, grad_storage_1)
        rnn_batch_layer_2.init_parameters_storage(model)
        rnn_batch_layer_3.init_parameters_storage(model)

        self.assertTrue(np.shares_memory(model, rnn_batch_layer_1.get_model()))
        self.assertTrue(np.shares_memory(rnn_batch_layer_1.get_model(), rnn_batch_layer_2.get_model()))

        out_hs = np.empty((batch_size, max_seq_length, dim_h), dtype=dtype)
        out_hs2 = np.empty((max_seq_length, batch_size, dim_h), dtype=dtype)
        out_hs_last2 = np.empty((batch_size, dim_h), dtype=dtype)
        for i in range(batch_size):
            rnn_layer.set_init_h(hs_init[i])
            out_hs[i, 0:seq_lengths[i]] = rnn_layer.forward(data_t[i, 0:seq_lengths[i]])
            rnn_batch_layer_2.set_init_h(np.reshape(hs_init[i], (1, dim_h)))
            # numpy trick to prevent 2nd singleton dimension from being squeezed out, integer array indexing and slicing
            dat = data[:, [i], :]  # remains (L, 1, D) instead of (L, D)
            out_hs2[:, [i]] = rnn_batch_layer_2.forward(dat, seq_lengths[[i]])
            out_hs_last2[i] = rnn_batch_layer_2.hs_last

        rnn_batch_layer_1.set_init_h(hs_init)
        out_hs1 = rnn_batch_layer_1.forward(data, seq_lengths)
        rnn_batch_layer_3.set_init_h(hs_init)
        out_hs3 = rnn_batch_layer_3.forward(data_t, seq_lengths)

        for i in range(batch_size):
            seq_length = seq_lengths[i]

            # check 0-padded hidden state for mini-batch versions
            if seq_length < max_seq_length:
                self.assertTrue(np.all(np.equal(out_hs1[seq_length:max_seq_length, i], 0.0)))
                self.assertTrue(np.all(np.equal(out_hs2[seq_length:max_seq_length, i], 0.0)))
                self.assertTrue(np.all(np.equal(out_hs3[i, seq_length:max_seq_length], 0.0)))

            # check non-0 hidden state for mini-batch versions
            hs_first = out_hs[i, 0:seq_length]
            self.assertTrue(np.allclose(hs_first, out_hs1[0:seq_length, i], rtol=tolerance, atol=tolerance))
            self.assertTrue(np.allclose(hs_first, out_hs2[0:seq_length, i], rtol=tolerance, atol=tolerance))
            self.assertTrue(np.allclose(hs_first, out_hs3[i, 0:seq_length], rtol=tolerance, atol=tolerance))

            # check last-hidden state
            hs_last = out_hs[i, seq_length - 1] if seq_length > 0 else hs_init[i]
            self.assertTrue(np.allclose(hs_last, rnn_batch_layer_1.hs_last[i], rtol=tolerance, atol=tolerance))
            self.assertTrue(np.allclose(hs_last, out_hs_last2[i], rtol=tolerance, atol=tolerance))
            self.assertTrue(np.allclose(hs_last, rnn_batch_layer_3.hs_last[i], rtol=tolerance, atol=tolerance))

        # backwards propagation

        delta_err_1 = rnn_batch_layer_1.backwards(delta_upper)

        accum_grad = np.zeros(rnn_layer.get_num_p(), dtype=dtype)
        accum_grad_2 = np.zeros(rnn_layer.get_num_p(), dtype=dtype)

        for i in range(batch_size):
            # need to forward propagate again because forward_batch(data_t) and back_propagation_batch(delta_upper_t)
            # remember and re-use last data_t
            rnn_layer.set_init_h(hs_init[i])
            rnn_layer.forward(data_t[i, 0:seq_lengths[i]])

            delta_err = rnn_layer.backwards(delta_upper_t[i, 0:seq_lengths[i]])
            accum_grad += rnn_layer.get_gradient()

            # first non-zero part should be identical to one retrieved from non-batch
            self.assertTrue(np.allclose(delta_err, delta_err_1[0:seq_lengths[i], i], rtol=tolerance, atol=tolerance))
            # second part, if exists, should be 0.0
            if seq_lengths[i] < max_seq_length:
                self.assertTrue(np.all(np.equal(delta_err_1[seq_lengths[i]:max_seq_length, i], 0.0)))

            # check the 2 mini-batch versions

            rnn_batch_layer_2.set_init_h(np.reshape(hs_init[i], (1, dim_h)))
            rnn_batch_layer_2.forward(data[:, [i], :], seq_lengths[[i]])

            delta_err_2 = rnn_batch_layer_2.backwards(delta_upper[:, [i], :])
            self.assertTrue(np.allclose(delta_err_1[:, [i], :], delta_err_2, rtol=tolerance, atol=tolerance))
            accum_grad_2 += rnn_batch_layer_2.get_gradient()

        self.assertTrue(np.allclose(accum_grad, rnn_batch_layer_1.get_gradient(), rtol=tolerance, atol=tolerance))
        self.assertTrue(np.allclose(accum_grad_2, rnn_batch_layer_1.get_gradient(), rtol=tolerance, atol=tolerance))
        self.assertTrue(np.all(np.equal(rnn_batch_layer_1.get_model(), rnn_batch_layer_1.get_built_model())))
        self.assertTrue(np.all(np.equal(rnn_batch_layer_1.get_gradient(), rnn_batch_layer_1.get_built_gradient())))
        self.assertTrue(np.shares_memory(grad_storage_1, rnn_batch_layer_1.get_gradient()))


if __name__ == "__main__":
    unittest.main()
