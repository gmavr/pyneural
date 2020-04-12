import unittest

import numpy as np

import pyneural.neural_layer as nl
import pyneural.test.gradient_check_test_shared as gcs
from pyneural.ce_l2_loss import LayerWithL2Loss


class TestNeuralLayer(gcs.GradientCheckTestShared):

    @staticmethod
    def create_random_data(dim_x, dim_y, dtype, num_params, num_samples):
        model = 0.1 * np.random.standard_normal(num_params).astype(dtype)
        x = np.random.standard_normal((num_samples, dim_x)).astype(dtype)
        y = np.random.standard_normal((num_samples, dim_y)).astype(dtype)
        return x, y, model

    def test_batch_single(self):
        num_samples = 10
        dim_x, dim_y = 4, 3
        dtype, tolerance = np.float64, 1e-15

        n_layer = nl.NeuralLayer(dim_x, dim_y, dtype, activation='sigmoid')

        np.random.seed(47)
        x, y, model = TestNeuralLayer.create_random_data(dim_x, dim_y, dtype, n_layer.get_num_p(), num_samples)
        delta_err = 0.2 * np.random.standard_normal((num_samples, dim_y)).astype(dtype)

        n_layer.init_parameters_storage(model=model)

        y1 = np.empty((num_samples, dim_y)).astype(dtype)

        dw1 = np.zeros((dim_y, dim_x), dtype)
        db1 = np.zeros((dim_y, ), dtype)
        delta_err_new1 = np.empty((num_samples, dim_x), dtype)
        for i in range(num_samples):
            y1[i] = n_layer.forward_single(x[i])
            dw_t, db_t, delta_err_new_t = n_layer.backwards_single(delta_err[i])
            dw1 += dw_t
            db1 += db_t
            delta_err_new1[i, :] = delta_err_new_t

        y2 = n_layer.forward(x)
        delta_err_new2 = n_layer.backwards(delta_err)

        # results from matrix operations should be identical, except that in ubuntu elements y1[9,1], y2[9,1] and
        # some elements in delta_err_new1 and delta_err_new2 differ
        self.assertLessEqual(np.amax(np.fabs(y1 - y2)), tolerance)
        self.assertLessEqual(np.amax(np.fabs(delta_err_new1 - delta_err_new2)), tolerance)

        # results from summations are equal with very high numerical precision
        self.assertLessEqual(np.amax(np.fabs(dw1 - n_layer.dw)), tolerance)
        self.assertLessEqual(np.amax(np.fabs(db1 - n_layer.db)), tolerance)

    def test_gradients(self):
        num_samples = 10
        dim_x, dim_y = 4, 3
        dtype, tolerance = np.float64, 1e-9

        n_layer = nl.NeuralLayer(dim_x, dim_y, dtype, activation='tanh')

        np.random.seed(47)
        x, y, model = TestNeuralLayer.create_random_data(dim_x, dim_y, dtype, n_layer.get_num_p(), num_samples)

        loss_and_layer = LayerWithL2Loss(n_layer)
        loss_and_layer.init_parameters_storage(model)

        self.do_param_gradient_check(loss_and_layer, x, y, tolerance)
        self.do_input_gradient_check(loss_and_layer, x, y, tolerance)

        # test projection layer only, no activation

        n_layer = nl.NeuralLayer(dim_x, dim_y, dtype, activation=None)
        loss_and_layer = LayerWithL2Loss(n_layer)
        loss_and_layer.init_parameters_storage(model)

        self.do_param_gradient_check(loss_and_layer, x, y, tolerance)
        self.do_input_gradient_check(loss_and_layer, x, y, tolerance)

    def test_model_updates(self):
        num_samples = 10
        dim_x, dim_y = 4, 3
        dtype = np.float64

        n_layer = nl.NeuralLayer(dim_x, dim_y, dtype, activation='tanh')

        np.random.seed(47)
        x = np.random.standard_normal((num_samples, dim_x)).astype(dtype)
        model = 0.1 * np.random.standard_normal(n_layer.get_num_p()).astype(dtype)

        grad = np.zeros(n_layer.get_num_p()).astype(dtype)
        n_layer.init_parameters_storage(model=model, grad=grad)

        n_layer.forward(x)

        self.assertTrue(np.alltrue(np.equal(model, n_layer.get_model())))
        self.assertTrue(np.shares_memory(model, n_layer.get_model()))

        model[dim_x:(dim_x + dim_y)] = 0.1 * np.random.standard_normal(dim_y).astype(dtype)

        # check that the model contents inside n_layer changed
        self.assertTrue(np.alltrue(np.equal(model, n_layer.get_model())))
        self.assertTrue(np.alltrue(np.equal(model, n_layer.get_built_model())))
        self.assertTrue(np.shares_memory(model, n_layer.get_model()))

        n_layer.forward(x)

        delta_upper = 0.1 * np.random.standard_normal((num_samples, dim_y)).astype(dtype)
        n_layer.backwards(delta_upper)

        self.assertTrue(np.alltrue(np.equal(grad, n_layer.get_built_gradient())))
        self.assertTrue(np.shares_memory(grad, n_layer.get_gradient()))


if __name__ == "__main__":
    unittest.main()
