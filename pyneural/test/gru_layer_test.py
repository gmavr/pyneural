import unittest

import numpy as np

import gradient_check_test_shared as gcs
import pyneural.gru_layer as gru
from pyneural.ce_l2_loss import LayerWithL2Loss


class TestGru(gcs.GradientCheckTestShared):

    @staticmethod
    def create_random_data(dim_x, dim_y, dtype, num_params, num_samples):
        model = 0.1 * np.random.standard_normal(num_params).astype(dtype)
        x = 0.5 * np.random.standard_normal((num_samples, dim_x)).astype(dtype)
        y = 0.5 * np.random.standard_normal((num_samples, dim_y)).astype(dtype)
        h_init = 0.1 * np.random.standard_normal(dim_y).astype(dtype)
        return x, y, model, h_init
    
    def test_gradients(self):
        num_samples = 13
        dim_x, dim_h = 7, 5
        dtype, tolerance = np.float64, 1e-9

        n_layer = gru.GruLayer(dim_x, dim_h, num_samples + 2, dtype)

        np.random.seed(seed=47)
        x, y, model, h_init = TestGru.create_random_data(dim_x, dim_h, dtype, n_layer.get_num_p(), num_samples)

        loss_and_layer = LayerWithL2Loss(n_layer)
        loss_and_layer.init_parameters_storage(model=model)

        self.do_param_gradient_check(loss_and_layer, x, y, tolerance, h_init)
        self.do_input_gradient_check(loss_and_layer, x, y, tolerance, h_init)

    def test_forward_equivalence(self):
        """Verifies that a batched invocation of N sequences and N non-batched invocations of 1 sequence return the same
         results.
        """
        dim_d, dim_h = 3, 2
        batch_size = 4
        max_batch_size = batch_size + 1
        max_seq_length = 4
        dtype, tolerance = np.float64, 1e-14

        # intentionally all sequences shorter than max_seq_length
        data_t = np.array([
            [[3.1, 0.1, -7.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.1, 2.1, -3.2], [2.1, -4.5, 3.4], [5.0, -3.1, -0.6], [0.0, 0.0, 0.0]],
            [[1.5, -0.7, 5.1], [-9.1, 2.7, 0.6], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[-5.0, 0.1, 0.2], [2.7, 9.1, -2.0], [2.1, -1.5, 1.4], [0.0, 0.0, 0.0]],
            ], dtype=dtype)
        seq_lengths = np.array([1, 3, 2, 3], dtype=np.int)

        self.assertEquals(data_t.shape, (batch_size, max_seq_length, dim_d))
        self.assertEquals(seq_lengths.shape, (batch_size, ))

        data = np.empty((max_seq_length, batch_size, dim_d), dtype=dtype)
        for i in range(max_seq_length):
            data[i] = np.copy(data_t[:, i, :])

        rnn_layer = gru.GruLayer(dim_d, dim_h, max_seq_length, dtype)
        rnn_batch_layer = gru.GruBatchLayer(dim_d, dim_h, max_seq_length, max_batch_size, dtype)

        np.random.seed(seed=47)
        model = 0.1 * np.random.standard_normal((rnn_layer.get_num_p(),)).astype(dtype)
        hs_init = 0.01 * np.random.standard_normal((batch_size, dim_h)).astype(dtype)

        # all following objects share the same model memory, but each has its own separate gradient memory
        rnn_layer.init_parameters_storage(model)
        rnn_batch_layer.init_parameters_storage(model)

        self.assertTrue(np.shares_memory(model, rnn_batch_layer.get_model()))

        out_hs = np.empty((max_batch_size, max_seq_length, dim_h), dtype=dtype)
        for i in range(batch_size):
            rnn_layer.set_init_h(hs_init[i])
            out_hs[i, 0:seq_lengths[i]] = rnn_layer.forward(data_t[i, 0:seq_lengths[i]])

        rnn_batch_layer.set_init_h(hs_init)
        out_hs_batch = rnn_batch_layer.forward(data, seq_lengths)

        for i in range(batch_size):
            seq_length = seq_lengths[i]

            # check 0-padded hidden state for batched version
            if seq_length < max_seq_length:
                # numpy returns bool_ which needs cast to bool
                self.assertTrue(bool(np.alltrue(np.equal(out_hs_batch[seq_length:max_seq_length, i], 0.0))))

            # check non-0 hidden state for batched version
            hs_first = out_hs[i, 0:seq_length]
            self.assertTrue(np.allclose(hs_first, out_hs_batch[0:seq_length, i], rtol=tolerance, atol=tolerance))

            # check last-hidden state
            hs_last = out_hs[i, seq_length - 1]
            hs_init[i] = hs_last
            self.assertTrue(np.allclose(hs_last, rnn_batch_layer.hs_last[i], rtol=tolerance, atol=tolerance))

        # one more forward propagation to verify that last hidden state is forwarded the same in both case
        for i in range(batch_size):
            rnn_layer.set_init_h(hs_init[i])
            out_hs[i, 0:seq_lengths[i]] = rnn_layer.forward(data_t[i, 0:seq_lengths[i]])

        out_hs_batch = rnn_batch_layer.forward(data, seq_lengths)

        for i in range(batch_size):
            seq_length = seq_lengths[i]

            # check non-0 hidden state for batched version
            hs_first = out_hs[i, 0:seq_length]
            self.assertTrue(np.allclose(hs_first, out_hs_batch[0:seq_length, i], rtol=tolerance, atol=tolerance))

            # check last-hidden state
            hs_last = out_hs[i, seq_length - 1]
            self.assertTrue(np.allclose(hs_last, rnn_batch_layer.hs_last[i], rtol=tolerance, atol=tolerance))


if __name__ == "__main__":
    unittest.main()
