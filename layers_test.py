import time
import unittest

import numpy as np

import ce_softmax_layer as ce_sm
import dataset as dst
import embedding_layer as em
import gradient_check as gc
import gradient_check_test_shared as gcs
import layers
import rnn_batch_layer as rb


""" Tests of various multi-layer networks involving RNNs.
"""


def create_rnn_random_data(rnn_obj, num_samples, input_vocab=None):
    dtype = rnn_obj.get_dtype()
    dim_d, dim_h, dim_k = rnn_obj.get_dimensions()

    # the difference in sd of normal initialization is not necessarily because of extensive trial and error but due to
    # arbitrary decisions that we stuck to for tracking effect of other changes
    if input_vocab is not None:
        model = 0.1 * np.random.standard_normal(rnn_obj.get_num_p()).astype(dtype)
        inputs = np.random.randint(0, input_vocab, num_samples)
    else:
        model = 0.01 * np.random.standard_normal(rnn_obj.get_num_p()).astype(dtype)
        inputs = np.random.standard_normal((num_samples, dim_d)).astype(dtype)
    labels = np.random.randint(0, dim_k, num_samples)

    h_init = 0.001 * np.random.standard_normal(dim_h).astype(dtype)
    return model, inputs, labels, h_init


class TestRnnLayer(gcs.GradientCheckTestShared):

    def test_rnn_softmax_normal_init(self):
        seq_length = 20
        bptt_steps = seq_length
        dim_d, dim_h, dim_k = 5, 7, 3

        rnn_obj = layers.RnnSoftMax((dim_d, dim_h, dim_k), seq_length, bptt_steps, dtype=np.float32)
        rnn_obj.init_parameters_storage()
        rnn_obj.model_normal_init(sd=0.1)
        params = rnn_obj.get_model()

        # after normal initialization, the bias terms should be set to 0
        self.assertEquals(np.count_nonzero(params), rnn_obj.get_num_p() - (dim_h + dim_k))

        # validate zero mean and std
        non_zero_bool = params != 0.0
        self.assertLess(np.abs(np.mean(params[non_zero_bool])), 0.02)
        self.assertLess(np.abs(np.std(params[non_zero_bool]) - 0.1), 0.05)

    def test_rnn_gradients(self):
        seq_length = 20
        bptt_steps = seq_length
        # bptt_steps != seq_length correctly fails gradient check
        dim_d, dim_h, dim_k = 5, 7, 3
        dtype, tolerance = np.float64, 1e-9

        rnn_obj = layers.RnnSoftMax((dim_d, dim_h, dim_k), seq_length, bptt_steps=bptt_steps, dtype=dtype)

        np.random.seed(seed=47)
        model, inputs, labels, h_init = create_rnn_random_data(rnn_obj, seq_length)

        rnn_obj.init_parameters_storage(model)

        self.do_param_gradient_check(rnn_obj, inputs, labels, tolerance, h_init)
        self.do_input_gradient_check(rnn_obj, inputs, labels, tolerance, h_init)

    def test_rnn_embedding_param_gradient(self):
        seq_length = 30
        bptt_steps = seq_length
        dim_d, dim_h, dim_k, dim_v = 7, 5, 23, 19
        dtype, tolerance = np.float64, 1e-8

        rnn_obj = layers.RnnSoftMax((dim_d, dim_h, dim_k), seq_length, bptt_steps=bptt_steps, dtype=dtype)
        em_obj = em.EmbeddingLayer(dim_k, dim_d, dtype)
        rnn_obj_em = layers.RnnEmbeddingsSoftMax(rnn_obj, em_obj)

        np.random.seed(seed=47)
        model, inputs, labels, h_init = create_rnn_random_data(rnn_obj_em, seq_length, dim_v)
        rnn_obj_em.init_parameters_storage(model)

        self.do_param_gradient_check(rnn_obj_em, inputs, labels, tolerance, h_init)

    def test_batched_rnn_embedding_param_gradient(self):
        max_batch_size = 5
        max_seq_length = 6
        dim_d, dim_h, dim_k, dim_v = 7, 5, 23, 19
        dtype, tolerance = np.float64, 1e-8

        em_batch = em.EmbeddingLayerBatch(dim_k, dim_d, max_seq_length, max_batch_size, dtype)
        rnn_batch = rb.RnnBatchLayer(dim_d, dim_h, max_seq_length, max_batch_size, dtype=dtype)
        ce_sm_batch = ce_sm.CESoftmaxLayerBatch(dim_k, dim_h, max_seq_length, max_batch_size, dtype)

        loss_batch = layers.RnnEmbeddingsSoftMaxBatch(ce_sm_batch, rnn_batch, em_batch)

        batch_size = max_batch_size - 1  # set smaller batch than maximum

        np.random.seed(seed=47)

        seq_lengths = np.random.randint(max_seq_length - 3, max_seq_length, batch_size)
        seq_lengths[0] = 0  # set first sequence to be 0 length
        x = np.zeros((max_seq_length, batch_size), dtype=np.int)
        labels = np.zeros((max_seq_length, batch_size), dtype=np.int)
        for j in range(batch_size):
            seq_length = seq_lengths[j]
            x[0:seq_length, j] = np.random.randint(0, dim_v, seq_length)
            labels[0:seq_length, j] = np.random.randint(0, dim_k, seq_length)

        model = 0.1 * np.random.standard_normal((loss_batch.get_num_p(),)).astype(dtype)
        h_init = 0.01 * np.random.standard_normal((batch_size, dim_h)).astype(dtype)

        loss_batch.init_parameters_storage(model)

        self.do_param_batched_gradient_check(loss_batch, x, labels, seq_lengths, tolerance, h_init)

    # FIXME: Test broke
    # def test_class_rnn_param_gradient(self):
    #     num_samples = 20
    #     dim_d, dim_h, dim_k = (4, 10, 8)
    #
    #     dtype, tolerance = (np.float64, 1e-8)
    #
    #     word_class_mapper = rnn.WordClassMapper(4, dim_k)
    #
    #     rnn_obj = rnn.RnnClassSoftMax((dim_d, dim_h, dim_k), word_class_mapper, num_samples, dtype=dtype)
    #
    #     np.random.seed(seed=47)
    #     params, inputs, labels, h_init = create_random_data(rnn_obj, num_samples)
    #
    #     self.do_param_gradient_check(rnn_obj, inputs, labels, params, tolerance, h_init)

    def test_class_rnn_input_gradient(self):
        num_samples = 5
        dim_d, dim_h, dim_k = 4, 10, 6

        dtype, tolerance = np.float64, 1e-9

        word_class_mapper = layers.WordClassMapper(2, dim_k)

        loss_nn = layers.RnnClassSoftMax((dim_d, dim_h, dim_k), word_class_mapper, num_samples, dtype=dtype)

        np.random.seed(seed=47)
        model, data, labels, h_init = create_rnn_random_data(loss_nn, num_samples)
        loss_nn.init_parameters_storage(model)

        self.do_input_gradient_check(loss_nn, data, labels, tolerance, h_init)

    @staticmethod
    def forward_backward_batch_with_init(rnn_and_data, h_init):
        """
        Helper function for gradient check
        """
        rnn_and_data.loss_nn.set_init_h(h_init)
        loss, grad = rnn_and_data.forward_backward_batch()
        return loss, grad

    def test_rnn_param_gradient_with_boundary_dataset(self):
        """
        Tests that the derivative is correct when parts of different sequences are placed together in the same
        mini-batch and therefore the hidden state is reset at the boundary(ies).
        """
        batch_size = 10
        bptt_steps = batch_size  # bptt_steps != batch_size correctly fails gradient check
        dim_d, dim_h, dim_k = 5, 10, 3
        dtype, tolerance = np.float64, 1e-8

        rnn_obj = layers.RnnSoftMax((dim_d, dim_h, dim_k), batch_size, bptt_steps=bptt_steps, dtype=dtype)

        np.random.seed(seed=47)
        model, inputs, labels, h_init = create_rnn_random_data(rnn_obj, batch_size)

        rnn_obj.init_parameters_storage(model)

        docs_end = np.array([4, 6, 7, batch_size-1])
        # this test code relies on the underlying InMemoryDataSetWithBoundaries object to return the same data at each
        # invocation (therefore samples in data set == batch_size)
        assert docs_end[-1] == batch_size - 1

        dataset = dst.InMemoryDataSetWithBoundaries(inputs, labels, docs_end)

        rnn_and_data = dst.LossNNAndDataSetWithBoundary(rnn_obj, dataset, batch_size)

        print("Starting parameters gradient check for: %s with %s. Number parameters: %d"
              % (rnn_obj.get_class_fq_name(), rnn_and_data.__class__.__name__, rnn_obj.get_num_p()))
        start_time = time.time()
        success = gc.gradient_check(lambda: self.forward_backward_batch_with_init(rnn_and_data, h_init),
                                    model, tolerance=tolerance)
        self.assertTrue(success)
        time_elapsed = (time.time() - start_time)

        num_invocations = (1 + 2 * rnn_obj.get_num_p())
        print("per invocation time: %.4g sec" % (time_elapsed / num_invocations))
        print("total time elapsed for %d invocations: %.4g sec" % (num_invocations, time_elapsed))


if __name__ == "__main__":
    unittest.main()
