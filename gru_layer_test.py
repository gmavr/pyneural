import unittest
import time

import numpy as np

from ce_l2_loss import LayerWithL2Loss
import gradient_check_test_shared as gcs
import gru_layer as gru


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

    def test_forward_batch(self):
        num_samples = 17
        dim_d, dim_h, max_seq_length = 5, 7, num_samples
        dtype, tolerance = np.float64, 1e-15

        gru1 = gru.GruLayer(dim_d, dim_h, max_seq_length, dtype)
        gru2 = gru.GruLayer(dim_d, dim_h, max_seq_length, dtype)

        np.random.seed(seed=47)
        x, _, model, _ = TestGru.create_random_data(dim_d, dim_h, dtype, gru1.get_num_p(), num_samples)

        gru1.init_parameters_storage(model=model)
        gru2.init_parameters_storage(model=np.copy(model))

        # Timing test is not reliable. For dim_d = 500, dim_h = 200, by reordering the following two forward
        # propagation calls, whichever version runs first usually takes longer to execute.
        # After observing many reorderings with large matrix sizes, the optimized version is a little faster (by 1%-10%)
        # as dimensions increase.

        # start_time = time.time()
        y1 = gru1.forward(x)
        # time_elapsed = (time.time() - start_time)
        # print("time: %.4g sec" % time_elapsed)

        # start_time = time.time()
        y2 = gru2.forward_batch_debug(x)
        # time_elapsed = (time.time() - start_time)
        # print("time: %.4g sec" % time_elapsed)

        self.assertTrue(np.allclose(y1, y2, rtol=1e-15, atol=1e-15))


if __name__ == "__main__":
    unittest.main()
