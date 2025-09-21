import unittest

import numpy as np

import pyneural.drop_out_layer as dl
import pyneural.neural_layer as nl
import pyneural.test.gradient_check_test_shared as gcs
from pyneural.neural_base import LossNN


class DoubleNeuralLayerWithDropoutL2Loss(LossNN):
    """
    This class consists of 2 neural layers and a scalar loss output loss function on top.
    It allows dropout to be set to inputs and after each layer.

    This is a good example for how to use drop-out.
    """

    def __init__(self, n_layer_1, n_layer_2, drop_p1, drop_p2, drop_p3, inverted_drop_out=True):
        assert isinstance(n_layer_1, nl.NeuralLayer)
        assert isinstance(n_layer_2, nl.NeuralLayer)
        assert n_layer_1.get_dtype() == n_layer_2.get_dtype()
        self.n_layer_1 = n_layer_1
        self.n_layer_2 = n_layer_2
        self.inverted_drop_out = inverted_drop_out
        if inverted_drop_out:
            self.drop_out_layer_1 = dl.InvertedDropoutLayer(drop_p1)
            self.drop_out_layer_2 = dl.InvertedDropoutLayer(drop_p2)
            self.drop_out_layer_3 = dl.InvertedDropoutLayer(drop_p3)
        else:
            self.drop_out_layer_1 = dl.DropoutLayer(drop_p1)
            self.drop_out_layer_2 = dl.DropoutLayer(drop_p2)
            self.drop_out_layer_3 = dl.DropoutLayer(drop_p3)

        self.fixed_mask = False

        self.delta_err_top = None
        self.loss = None

        num_p = self.n_layer_1.get_num_p() + self.n_layer_2.get_num_p()
        super().__init__(num_p, n_layer_1.get_dtype())

    def get_display_dict(self):
        d = self._init_display_dict()
        d['drop_out_layer_1'] = self.drop_out_layer_1.get_display_dict()
        d['drop_out_layer_2'] = self.drop_out_layer_2.get_display_dict()
        d['drop_out_layer_3'] = self.drop_out_layer_3.get_display_dict()
        d['n_layer_1'] = self.n_layer_1.get_display_dict()
        d['n_layer_2'] = self.n_layer_2.get_display_dict()
        return d

    def model_glorot_init(self):
        self.n_layer_1.model_glorot_init()
        self.n_layer_2.model_glorot_init()

    def forward_batch_train(self, x, y):
        if self.fixed_mask:
            x = self.drop_out_layer_1.forward_train_with_fixed_mask(x)
            x = self.n_layer_1.forward(x)
            x = self.drop_out_layer_2.forward_train_with_fixed_mask(x)
            x = self.n_layer_2.forward(x)
            x = self.drop_out_layer_3.forward_train_with_fixed_mask(x)
        else:
            x = self.drop_out_layer_1.forward_train(x)
            x = self.n_layer_1.forward(x)
            x = self.drop_out_layer_2.forward_train(x)
            x = self.n_layer_2.forward(x)
            x = self.drop_out_layer_3.forward_train(x)

        self.delta_err_top = x - y  # (N, Dy)

        # By definition of dropout the dropped elements literally do not exist in the network.
        # So the zeroed-out output elements of last layer must also be removed from consideration from both the loss
        # function and the loss back-propagated error
        mask = self.drop_out_layer_3.get_drop_out_mask()

        self.delta_err_top *= mask
        self.loss = 0.5 * np.sum(self.delta_err_top * self.delta_err_top)

    def forward_batch_eval(self, x):
        x = self.drop_out_layer_1.forward_eval(x)
        x = self.n_layer_1.forward(x)
        x = self.drop_out_layer_2.forward_eval(x)
        x = self.n_layer_2.forward(x)
        return self.drop_out_layer_3.forward_eval(x)

    def back_propagation_batch(self):
        delta_err_l = self.drop_out_layer_3.backwards(self.delta_err_top)
        delta_err_l = self.n_layer_2.backwards(delta_err_l)
        # inline equivalent: delta_err_l *= .drop_out_layer_2._drop_out_mask
        delta_err_l = self.drop_out_layer_2.backwards(delta_err_l)
        delta_err_l = self.n_layer_1.backwards(delta_err_l)
        # inline equivalent: delta_err_l *= .drop_out_layer_1._drop_out_mask
        delta_err_l = self.drop_out_layer_1.backwards(delta_err_l)
        return delta_err_l

    def forward_backwards(self, x, y_true):
        self._x, self._y_true = x, y_true

        self.forward_batch_train(x, y_true)
        delta_err = self.back_propagation_batch()
        return self.loss, self._grad, delta_err

    def _set_model_references_in_place(self):
        params = self._model
        nl_num_p = self.n_layer_1.get_num_p()
        self.n_layer_1.set_model_storage(params[0:nl_num_p])
        self.n_layer_2.set_model_storage(params[nl_num_p:])

    def _set_gradient_references_in_place(self):
        grad = self._grad
        nl_num_p = self.n_layer_1.get_num_p()
        self.n_layer_1.set_gradient_storage(grad[0:nl_num_p])
        self.n_layer_2.set_gradient_storage(grad[nl_num_p:])

    def get_built_model(self):
        return np.concatenate((self.n_layer_1.get_model(), self.n_layer_2.get_model()))

    def get_built_gradient(self):
        return np.concatenate((self.n_layer_1.get_gradient(), self.n_layer_2.get_gradient()))


class TestDropoutLayer(gcs.GradientCheckTestShared):

    @staticmethod
    def create_random_data(dim_x, dim_y, dtype, num_params, num_samples):
        model = np.random.standard_normal(num_params).astype(dtype)
        x = np.random.uniform(-1.0, 1.0, size=(num_samples, dim_x)).astype(dtype)
        y = np.random.uniform(-1.0, 1.0, size=(num_samples, dim_y)).astype(dtype)
        return x, y, model

    def test_gradients(self):
        self.verify_gradients(inverted_drop_out=True)
        self.verify_gradients(inverted_drop_out=False)

    def verify_gradients(self, inverted_drop_out):
        batch_size = 20
        dim_x, dim_z, dim_y = 6, 5, 8
        drop_p1, drop_p2, drop_p3 = 0.6, 0.5, 0.4

        dtype, tolerance = (np.float64, 5e-8)

        n_layer_1 = nl.NeuralLayer(dim_x, dim_z, dtype, 'tanh')
        n_layer_2 = nl.NeuralLayer(dim_z, dim_y, dtype, 'tanh')

        loss_nn = DoubleNeuralLayerWithDropoutL2Loss(n_layer_1, n_layer_2, drop_p1, drop_p2, drop_p3,
                                                     inverted_drop_out=inverted_drop_out)

        np.random.seed(47)
        x, y, model = TestDropoutLayer.create_random_data(dim_x, dim_y, dtype, loss_nn.get_num_p(), batch_size)

        loss_nn.init_parameters_storage(model=model)

        # for gradient check to work drop-out masks have to be remain constant, so they are set up here
        loss_nn.fixed_mask = True
        drop_out_mask_1 = np.random.binomial(1, drop_p1, size=(batch_size, dim_x)).astype(dtype)
        loss_nn.drop_out_layer_1.set_drop_out_mask(drop_out_mask_1)
        drop_out_mask_2 = np.random.binomial(1, drop_p2, size=(batch_size, dim_z)).astype(dtype)
        loss_nn.drop_out_layer_2.set_drop_out_mask(drop_out_mask_2)
        drop_out_mask_3 = np.random.binomial(1, drop_p3, size=(batch_size, dim_y)).astype(dtype)
        loss_nn.drop_out_layer_3.set_drop_out_mask(drop_out_mask_3)

        self.do_param_gradient_check(loss_nn, x, y, tolerance)
        self.do_input_gradient_check(loss_nn, x, y, tolerance)

    def test_distribution(self):
        self.verify_distribution(True)
        self.verify_distribution(False)

    def verify_distribution(self, inverted_drop_out):
        batch_size = 1000
        dim_x, dim_z, dim_y = 20, 40, 30
        drop_p1, drop_p2, drop_p3 = 0.7, 0.2, 0.8

        dtype = np.float32

        n_layer_1 = nl.NeuralLayer(dim_x, dim_z, dtype, 'tanh')
        n_layer_2 = nl.NeuralLayer(dim_z, dim_y, dtype, 'tanh')

        loss_nn = DoubleNeuralLayerWithDropoutL2Loss(n_layer_1, n_layer_2, drop_p1, drop_p2, drop_p3,
                                                     inverted_drop_out=inverted_drop_out)

        np.random.seed(47)
        x, y, _ = TestDropoutLayer.create_random_data(dim_x, dim_y, dtype, loss_nn.get_num_p(), batch_size)
        x += 1  # center at 1
        x1 = np.copy(x)

        loss_nn.init_parameters_storage()
        loss_nn.model_glorot_init()

        loss_nn.forward_batch_train(x, y)
        assert drop_p1 - 0.1 < 1 - float(np.count_nonzero(loss_nn.n_layer_1.x)) / (dim_x * batch_size) < drop_p1 + 0.1
        m1, sd1 = np.mean(loss_nn.n_layer_1.x), np.std(loss_nn.n_layer_1.x)
        sd2 = np.std(loss_nn.n_layer_2.x)
        if inverted_drop_out:
            sd3 = np.std(loss_nn.drop_out_layer_3.get_y())
        else:
            sd3 = np.std(loss_nn.n_layer_2.y)

        loss_nn.forward_batch_eval(x1)

        m1a, sd1a = np.mean(loss_nn.n_layer_1.x), np.std(loss_nn.n_layer_1.x)
        sd2a = np.std(loss_nn.n_layer_2.x)
        if inverted_drop_out:
            sd3a = np.std(loss_nn.drop_out_layer_3.get_y())
        else:
            sd3a = np.std(loss_nn.n_layer_2.y)

        # inputs are centered at 1 (before drop-out in train and scaling in eval)
        self.assertTrue(0.9 < m1 / m1a < 1.1)
        # the mean outputs of the rest of the layers are close too

        # the variance with training dropout should be much higher than with evaluation dropout because the former has
        # values zeroes while the latter has non-zero values scaled.
        self.assertTrue(sd1 > 2 * sd1a)
        self.assertTrue(sd2 > 1.2 * sd2a)
        self.assertTrue(sd3 > 2 * sd3a)


if __name__ == "__main__":
    unittest.main()
