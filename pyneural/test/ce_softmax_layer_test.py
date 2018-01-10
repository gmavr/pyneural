import unittest

import numpy as np

import gradient_check_test_shared as gcs
import pyneural.ce_softmax_layer as ce_sm


class TestCESoftmaxLayer(gcs.GradientCheckTestShared):

    @staticmethod
    def create_random_data(ce_sm_layer, num_samples):
        dim_x, dim_k = ce_sm_layer.dim_d, ce_sm_layer.dim_k
        dtype, num_params = ce_sm_layer.get_dtype(), ce_sm_layer.get_num_p()

        np.random.seed(seed=47)
        x = np.random.standard_normal((num_samples, dim_x)).astype(dtype)
        model = 0.1 * np.random.standard_normal(num_params).astype(dtype)
        labels = np.random.randint(0, dim_k, num_samples)

        return labels, model, x

    def test_batch_single(self):
        num_samples = 10
        dim_x, dim_k = 5, 3
        dtype, rtol, atol = np.float64, 1e-15, 1e-15

        ce_sm_layer = ce_sm.CESoftmaxLayer(dim_k, dim_x, dtype)

        labels, params, x = TestCESoftmaxLayer.create_random_data(ce_sm_layer, num_samples)

        ce_sm_layer.init_parameters_storage(params)

        loss1 = np.empty((num_samples,)).astype(dtype)

        grad1 = np.zeros(ce_sm_layer.get_num_p(), dtype)
        delta_err_new1 = np.empty((num_samples, dim_x), dtype)
        for i in range(num_samples):
            loss1[i] = ce_sm_layer.forward_single(x[i], labels[i])
            delta_err_new1[i, :] = ce_sm_layer.backwards_single()
            grad = ce_sm_layer.get_gradient()
            grad1 += grad

        loss2 = ce_sm_layer.forward(x, labels)
        delta_err_new2 = ce_sm_layer.backwards()

        loss1a = np.sum(loss1)

        # results from summations are equal with very high numerical precision
        self.assertTrue(np.allclose(loss1a, loss2, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(delta_err_new1, delta_err_new2, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(grad1, ce_sm_layer.get_gradient(), rtol=rtol, atol=atol))

    def test_gradients(self):
        num_samples = 10
        dim_x, dim_k = 11, 13
        dtype, tolerance = np.float64, 1e-8

        ce_sm_layer = ce_sm.CESoftmaxLayer(dim_k, dim_x, dtype)

        labels, params, x = TestCESoftmaxLayer.create_random_data(ce_sm_layer, num_samples)
        ce_sm_layer.init_parameters_storage(params)

        self.do_param_gradient_check(ce_sm_layer, x, labels, tolerance)
        self.do_input_gradient_check(ce_sm_layer, x, labels, tolerance)

    def test_gradients_batched(self):
        batch_size = 5
        max_seq_length = 10
        dim_x, dim_k = 11, 13
        dtype, tolerance = np.float64, 1e-8

        ce_batch = ce_sm.CESoftmaxLayerBatch(dim_k, dim_x, max_seq_length, batch_size, dtype)
        ce_batch.init_parameters_storage()

        x = np.zeros((max_seq_length, batch_size, dim_x), dtype=dtype)

        np.random.seed(seed=47)
        ce_batch.model_normal_init(sd=0.1)

        # set first sequence to 0 length to test that case
        # set all other sequences to random lengths greater than 0 but smaller than the maximum allowed length that the
        # layer was initialized with
        seq_lengths = np.random.randint(max_seq_length - 3, max_seq_length, batch_size)
        seq_lengths[0] = 0
        for j in range(batch_size):
            x[0:seq_lengths[j], j, :] = 0.1 * np.random.standard_normal((seq_lengths[j], dim_x)).astype(dtype)

        labels = np.zeros((max_seq_length, batch_size), dtype=np.int)
        for j in range(batch_size):
            labels[0:seq_lengths[j], j] = np.random.randint(0, dim_k, seq_lengths[j])

        self.do_param_batched_gradient_check(ce_batch, x, labels, seq_lengths, tolerance)

        # the gradient check verifying derivative w.r.to inputs requires the batch to contain full sequences
        seq_lengths = np.empty((batch_size, ), dtype=np.int)
        seq_lengths.fill(max_seq_length)
        x = 0.1 * np.random.standard_normal((max_seq_length, batch_size, dim_x)).astype(dtype)

        labels = np.random.randint(0, dim_k, (max_seq_length, batch_size))
        for j in range(batch_size):
            labels[0:max_seq_length, j] = np.random.randint(max_seq_length)

        self.do_param_batched_gradient_check(ce_batch, x, labels, seq_lengths, tolerance)
        self.do_input_batched_gradient_check(ce_batch, x, labels, seq_lengths, tolerance)

    def test_forward_backward_equivalence(self):
        """ Verifies that CESoftmaxLayer and CESoftmaxLayerBatch behave identically """
        dim_d, dim_k = 3, 5
        batch_size = 5
        max_seq_length = 4

        dtype, rtol, atol = np.float64, 1e-14, 1e-14

        # intentionally all sequences shorter than max_seq_length
        data = np.array([
            [[3.1, 0.1, -7.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.1, 2.1, -3.2], [2.1, -4.5, 3.4], [5.0, -3.1, -0.6], [0.0, 0.0, 0.0]],
            [[1.5, -0.7, 5.1], [-9.1, 2.7, 0.6], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[-5.0, 0.1, 0.2], [2.7, 9.1, -2.0], [2.1, -1.5, 1.4], [0.0, 0.0, 0.0]],
            [[1.7, 2.4, -5.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ], dtype=dtype)
        seq_lengths = np.array([1, 3, 2, 3, 1], dtype=np.int)

        self.assertEquals(data.shape, (batch_size, max_seq_length, dim_d))
        self.assertEquals(seq_lengths.shape, (batch_size,))

        data2 = np.empty((max_seq_length, batch_size, dim_d), dtype=dtype)
        for i in range(max_seq_length):
            data2[i] = np.copy(data[:, i, :])

        labels = np.array([
            [1, 0, 2, 1, 4],
            [0, 4, 0, 0, 0],
            [0, 3, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ])

        self.assertEquals(labels.shape, (max_seq_length, batch_size))

        ce = ce_sm.CESoftmaxLayer(dim_k, dim_d, dtype)
        ce.init_parameters_storage()

        np.random.seed(seed=47)
        ce.model_normal_init(0.1)

        ce_batch = ce_sm.CESoftmaxLayerBatch(dim_k, dim_d, max_seq_length, batch_size, dtype)

        # we do not need to allocate our own gradient buffer for testing back propagation, so this is an additional
        # higher level test
        grad_storage = np.empty((ce_batch.get_num_p(),), dtype=dtype)
        ce_batch.init_parameters_storage(model=np.copy(ce.get_model()), grad=grad_storage)

        loss = 0.0
        p_hat = np.empty((max_seq_length, batch_size, dim_k), dtype=dtype)
        for j in range(batch_size):
            seq_length = seq_lengths[j]
            if seq_length > 0:
                loss += ce.forward(data2[0:seq_length, j, :], labels[0:seq_length, j])
                p_hat[0:seq_length, j, :] = np.copy(ce.p_hat)

        loss_batched = ce_batch.forward(data2, labels, seq_lengths)

        self.assertTrue(np.allclose(loss_batched, loss, rtol=rtol, atol=atol))

        for j in range(batch_size):
            seq_length = seq_lengths[j]
            if seq_length > 0:
                self.assertTrue(np.allclose(ce_batch.p_hat[0:seq_length, j], p_hat[0:seq_length, j],
                                            rtol=rtol, atol=atol))

        # check back propagation

        delta_err_t = ce_batch.backwards()

        accum_grad = np.zeros(ce.get_num_p(), dtype=dtype)
        for i in xrange(batch_size):
            # need to forward propagate again because forward_batch(data) and back_propagation_batch(delta_upper2)
            # remember and re-use last data
            ce.forward(data2[xrange(0, seq_lengths[i]), i], labels[xrange(0, seq_lengths[i]), i])
            d1 = ce.backwards()
            accum_grad += ce.get_gradient()
            # first non-zero part should be identical to one retrieved from non-batch
            self.assertTrue(np.allclose(d1, delta_err_t[0:seq_lengths[i], i], rtol=rtol, atol=atol))
            # second part, if exists, should be 0.0
            if seq_lengths[i] < max_seq_length:
                self.assertTrue(np.alltrue(np.equal(delta_err_t[seq_lengths[i]:max_seq_length, i], 0.0)))

        self.assertTrue(np.allclose(accum_grad, ce_batch.get_gradient(), rtol=rtol, atol=atol))
        self.assertTrue(np.alltrue(np.equal(ce_batch.get_model(), ce_batch.get_built_model())))
        self.assertTrue(np.alltrue(np.equal(ce_batch.get_gradient(), ce_batch.get_built_gradient())))
        self.assertTrue(np.shares_memory(grad_storage, ce_batch.get_gradient()))


if __name__ == "__main__":
    unittest.main()
