import unittest

import numpy as np

import gradient_check_test_shared as gcs
from layers import BidirRnnSoftMax, EmbeddingBidirRnnSoftMax


def create_random_data_dense_inputs(loss_nn, num_samples):
    model, labels, h_init_f = _create_random_data(loss_nn, num_samples)
    data = 0.1 * np.random.standard_normal((num_samples, loss_nn.dim_d)).astype(loss_nn.get_dtype())
    return model, data, labels, h_init_f


def create_random_data_embedding_inputs(loss_nn, num_samples):
    model, labels, h_init_f = _create_random_data(loss_nn, num_samples)
    data = np.random.randint(0, loss_nn.word_vocab_size, size=num_samples, dtype=np.int)
    return model, data, labels, h_init_f


def _create_random_data(loss_nn, num_samples):
    dtype = loss_nn.get_dtype()
    num_p = loss_nn.get_num_p()
    model = 0.1 * np.random.standard_normal(num_p).astype(dtype)
    labels = np.random.randint(0, loss_nn.dim_k, num_samples)
    h_init_f = 0.001 * np.random.standard_normal(loss_nn.dim_h).astype(dtype)
    return model, labels, h_init_f


class TestBidirRnnSoftMax(gcs.GradientCheckTestShared):

    def test_gradients(self):
        num_samples = 10
        dim_d, dim_h, dim_k = 5, 7, 3
        dimensions = (dim_d, dim_h, dim_k)
        dtype, tolerance = np.float64, 1e-9

        loss_nn = BidirRnnSoftMax(dimensions, num_samples, dtype=dtype, cell_type="basic")

        self.assertEqual(BidirRnnSoftMax.get_num_p_static(dim_d, dim_h, dim_k), loss_nn.get_num_p())

        np.random.seed(seed=47)
        model, data, labels, h_init = create_random_data_dense_inputs(loss_nn, num_samples)

        loss_nn.init_parameters_storage(model)

        self.do_param_gradient_check(loss_nn, data, labels, tolerance, h_init)
        self.do_input_gradient_check(loss_nn, data, labels, tolerance, h_init)

    def test_gradients_embeddings(self):
        num_samples = 10
        dim_v, dim_d, dim_h, dim_k = 4, 5, 10, 3
        dimensions = (dim_v, dim_d, dim_h, dim_k)
        dtype, tolerance = np.float64, 1e-8

        loss_nn = EmbeddingBidirRnnSoftMax(dimensions, num_samples, dtype=dtype)

        np.random.seed(seed=47)
        model, data, labels, h_init = create_random_data_embedding_inputs(loss_nn, num_samples)

        loss_nn.init_parameters_storage(model)

        self.do_param_gradient_check(loss_nn, data, labels, tolerance, h_init)


if __name__ == "__main__":
    unittest.main()
