import unittest

import numpy as np

import gradient_check_test_shared as gcs
import pyneural.ce_l2_loss as ce_l2_loss
import pyneural.embedding_layer as em


def create_random_data(em_object, num_samples):
    dtype = em_object.get_dtype()
    dim_k, dim_d = em_object.dim_k, em_object.dim_d

    x = np.random.randint(0, dim_k, num_samples)

    y_true = np.empty((num_samples, dim_d), dtype=dtype)
    for i in range(num_samples):
        # "auto-encoder" with added random noise
        y_true[i] = em_object.embedding_matrix[x[i]] + 0.01 * np.random.standard_normal(dim_d).astype(dtype)

    return x, y_true


def create_random_data_batch(em_object, max_seq_length, batch_size):
    """
    Args:
        em_object: embedding matrix MUST have been initialized for y_true to be set properly
        max_seq_length: maximum sequence length (first dimension)
        batch_size: number of sequences in batch (second dimension)

    Returns:
        x: (T, B) np.int
        y_true: (T, B, D) uses em_object.embedding_matrix
        delta_err: (T, B, D)
        seq_lengths: (B, ) np.int
    """
    # here we generate delta_err and y_true completely unrelated, so they can't be be used together (they are normally
    # connected by the loss function and predictions)
    dtype = em_object.get_dtype()
    dim_k, dim_d = em_object.dim_k, em_object.dim_d

    x = np.zeros((max_seq_length, batch_size), dtype=np.int)
    delta_err = np.zeros((max_seq_length, batch_size, dim_d), dtype=dtype)
    y_true = np.zeros((max_seq_length, batch_size, dim_d), dtype=dtype)

    # set first sequence to 0 length to test that case
    # set all other sequences to random lengths greater than 0 but smaller than the maximum allowed length that the
    # layer was initialized with
    seq_lengths = np.random.randint(1, max_seq_length, batch_size)
    seq_lengths[0] = 0
    for j in xrange(batch_size):
        seq_length = seq_lengths[j]
        if seq_length > 0:
            x[0:seq_length, j] = np.random.randint(0, dim_k, seq_length)
            delta_err[0:seq_length, j, :] = 0.01 * np.random.standard_normal((seq_length, dim_d)).astype(dtype)
            # "auto-encoder" with added random noise
            y_true[0:seq_length, j] = em_object.embedding_matrix[x[0:seq_length, j]] \
                + 0.01 * np.random.standard_normal((seq_length, dim_d)).astype(dtype)

    return x, y_true, delta_err, seq_lengths


class TestEmbeddingLayerGradients(gcs.GradientCheckTestShared):

    def test_forward_backward_equivalence(self):
        max_seq_length, batch_size = 5, 10
        dim_k, dim_d = 100, 20

        dtype, tolerance = (np.float64, 1e-13)

        em_obj = em.EmbeddingLayer(dim_k, dim_d, dtype)
        em_obj.init_parameters_storage()

        em_obj.model_normal_init(0.1)

        em_obj_batch = em.EmbeddingLayerBatch(dim_k, dim_d, max_seq_length, batch_size, dtype)
        em_obj_batch.init_parameters_storage(np.copy(em_obj.get_model()))

        x, _, delta_upper, seq_lengths = create_random_data_batch(em_obj_batch, max_seq_length, batch_size)

        out = np.empty((max_seq_length, batch_size, dim_d), dtype=dtype)
        for i in range(batch_size):
            out[range(0, seq_lengths[i]), i] = em_obj.forward(x[range(0, seq_lengths[i]), i])

        out_batch = em_obj_batch.forward(x, seq_lengths)

        for i in range(batch_size):
            seq_length = seq_lengths[i]
            if seq_length != 0:
                # check that the single sequence and the batch version return the same
                self.assertTrue(np.alltrue(np.equal(out[0:seq_length, i], out_batch[0:seq_length, i])))
            # check 0-padding
            self.assertTrue(np.alltrue(np.equal(out_batch[seq_length:max_seq_length, i], 0.0)))

        em_obj_batch.backwards(delta_upper)

        accum_grad = np.zeros(em_obj.get_num_p(), dtype=dtype)
        for i in xrange(batch_size):
            # need to forward propagate again because forward_batch(x) and back_propagation_batch(delta_upper)
            # remember and re-use last x
            em_obj.forward(x[0:seq_lengths[i], i])
            em_obj.backwards(delta_upper[0:seq_lengths[i], i])
            accum_grad += em_obj.get_gradient()

        self.assertTrue(np.allclose(accum_grad, em_obj_batch.get_gradient(), rtol=tolerance, atol=tolerance))
        self.assertTrue(np.alltrue(np.equal(em_obj_batch.get_model(), em_obj_batch.get_built_model())))
        self.assertTrue(np.alltrue(np.equal(em_obj_batch.get_gradient(), em_obj_batch.get_built_gradient())))

    def test_gradient_sparse_samples(self):
        num_samples = 10
        dim_k, dim_d = 15, 20
        dtype, tolerance = np.float64, 1e-12

        em_obj = em.EmbeddingLayer(dim_k, dim_d, dtype)
        assert num_samples < em_obj.dk_threshold

        loss_nn = ce_l2_loss.LayerWithL2Loss(em_obj)

        loss_nn.init_parameters_storage()

        em_obj.model_normal_init(0.01)

        model = em_obj.get_model()

        # validate zero mean and std
        self.assertLess(np.abs(np.mean(model)), 0.002)
        self.assertLess(np.abs(np.std(model) - 0.01), 0.005)

        x, y_true = create_random_data(em_obj, num_samples)

        self.do_param_gradient_check(loss_nn, x, y_true, tolerance)

    def test_gradient_dense_samples(self):
        num_samples = 40
        dim_k, dim_d = 10, 5
        dtype, tolerance = np.float64, 1e-12

        em_obj = em.EmbeddingLayer(dim_k, dim_d, dtype)
        assert num_samples > em_obj.dk_threshold

        loss_nn = ce_l2_loss.LayerWithL2Loss(em_obj)

        loss_nn.init_parameters_storage()

        em_obj.model_normal_init(0.01)

        x, y_true = create_random_data(em_obj, num_samples)

        self.do_param_gradient_check(loss_nn, x, y_true, tolerance)

    def test_gradient_batched(self):
        max_seq_length, batch_size = 5, 10
        dim_k, dim_d = 60, 20
        dtype, tolerance = np.float64, 1e-10

        em_obj_batch = em.EmbeddingLayerBatch(dim_k, dim_d, max_seq_length, batch_size, dtype)
        loss_and_layer = ce_l2_loss.BatchSequencesWithL2Loss(em_obj_batch)
        loss_and_layer.init_parameters_storage()

        em_obj_batch.model_normal_init(0.1)
        # create data set to contain fewer sequences than the batch size
        x, y_true, _, seq_lengths = create_random_data_batch(em_obj_batch, max_seq_length, batch_size-2)
        # make sure that we are testing the case of a 0 length sequence
        assert seq_lengths[0] == 0

        model = em_obj_batch.get_model()
        # validate zero mean and std
        self.assertLess(np.abs(np.mean(model)), 0.005)
        self.assertLess(np.abs(np.std(model) - 0.1), 0.005)

        self.do_param_batched_gradient_check(loss_and_layer, x, y_true, seq_lengths, tolerance)

    def test_gradient_manual(self):
        num_samples = 5
        dim_k, dim_d = 20, 3
        dtype = np.float32

        delta_err = np.random.standard_normal((num_samples, dim_d)).astype(dtype)
        x = np.array([1, 0, 1, 19, 8], dtype=np.int)

        em_layer = em.EmbeddingLayer(dim_k, dim_d, dtype)
        em_layer.init_parameters_storage()
        em_layer.model_normal_init(sd=1.0)

        out = em_layer.forward(x)
        self.assertEqual(out.shape, (num_samples, dim_d))

        em_layer.backwards(delta_err)
        gradient = em_layer.get_gradient()

        # reshape for more convenient indexing
        gradient1 = np.reshape(gradient, (dim_k, dim_d))

        self.assertTrue(np.alltrue(np.equal(gradient1[0], delta_err[1])))
        self.assertTrue(np.alltrue(np.equal(gradient1[1], delta_err[0] + delta_err[2])))
        self.assertTrue(np.alltrue(np.equal(gradient1[2], np.zeros((dim_d, )))))
        self.assertTrue(np.alltrue(np.equal(gradient1[8], delta_err[4])))
        self.assertTrue(np.alltrue(np.equal(gradient1[19], delta_err[3])))


if __name__ == "__main__":
    np.random.seed(seed=47)
    unittest.main()
