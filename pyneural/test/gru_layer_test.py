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


if __name__ == "__main__":
    unittest.main()
