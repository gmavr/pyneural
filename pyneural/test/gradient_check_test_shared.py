import time
import unittest

import numpy as np

import gradient_check as gc


class GradientCheckTestShared(unittest.TestCase):

    def do_param_gradient_check(self, loss_obj, x, y_true, tolerance, h_init=None):
        loss_obj.forward_backwards(x, y_true)  # needed only for setting y_true (should be a separate call)

        print("Starting model gradient check for: %s. Number parameters: %d"
              % (loss_obj.get_class_fq_name(), loss_obj.get_num_p()))
        start_time = time.time()
        if h_init is None:
            success = gc.gradient_check(loss_obj.forward_backwards_grad_model, loss_obj.get_model(), tolerance)
        else:
            success = gc.gradient_check(lambda: loss_obj.forward_backwards_grad_model(h_init=h_init),
                                        loss_obj.get_model(), tolerance)
        self.assertTrue(success)
        time_elapsed = (time.time() - start_time)

        num_invocations = (1 + 2 * loss_obj.get_num_p())
        print("per invocation time: %.4g sec" % (time_elapsed / num_invocations))
        print("total time elapsed for %d invocations: %.4g sec" % (num_invocations, time_elapsed))

        self.assertTrue(np.alltrue(np.equal(loss_obj.get_model(), loss_obj.get_built_model())))
        self.assertTrue(np.alltrue(np.equal(loss_obj.get_gradient(), loss_obj.get_built_gradient())))

    def do_input_gradient_check(self, loss_obj, x, y_true, tolerance, h_init=None):
        assert x.ndim == 2
        num_input_params = x.shape[0] * x.shape[1]

        loss_obj.forward_backwards(x, y_true)  # needed only for setting y_true (should be a separate call)

        print("Starting inputs parameters gradient check for: %s. Number parameters: %d"
              % (loss_obj.get_class_fq_name(), num_input_params))
        start_time = time.time()
        if h_init is None:
            success = gc.gradient_check(loss_obj.forward_backwards_grad_input, x, tolerance)
        else:
            success = gc.gradient_check(lambda: loss_obj.forward_backwards_grad_input(h_init=h_init), x, tolerance)
        self.assertTrue(success)
        time_elapsed = (time.time() - start_time)

        num_invocations = 1 + 2 * num_input_params
        print("per invocation time: %.4g sec" % (time_elapsed / num_invocations))
        print("total time elapsed for %d invocations: %.4g sec" % (num_invocations, time_elapsed))

        self.assertTrue(np.alltrue(np.equal(loss_obj.get_model(), loss_obj.get_built_model())))
        self.assertTrue(np.alltrue(np.equal(loss_obj.get_gradient(), loss_obj.get_built_gradient())))

    def do_param_batched_gradient_check(self, loss_obj, x, y_true, seq_lengths, tolerance, h_init=None):
        # needed only for setting y_true (should be a separate call)
        loss_obj.forward_backwards(x, y_true, seq_lengths)

        print("Starting model gradient check for: %s. Number parameters: %d"
              % (loss_obj.get_class_fq_name(), loss_obj.get_num_p()))
        start_time = time.time()
        if h_init is None:
            success = gc.gradient_check(loss_obj.forward_backwards_grad_model, loss_obj.get_model(), tolerance)
        else:
            success = gc.gradient_check(lambda: loss_obj.forward_backwards_grad_model(h_init=h_init),
                                        loss_obj.get_model(), tolerance)
        self.assertTrue(success)
        time_elapsed = (time.time() - start_time)

        num_invocations = (1 + 2 * loss_obj.get_num_p())
        print("per invocation time: %.4g sec" % (time_elapsed / num_invocations))
        print("total time elapsed for %d invocations: %.4g sec" % (num_invocations, time_elapsed))

        self.assertTrue(np.alltrue(np.equal(loss_obj.get_model(), loss_obj.get_built_model())))
        self.assertTrue(np.alltrue(np.equal(loss_obj.get_gradient(), loss_obj.get_built_gradient())))

    def do_input_batched_gradient_check(self, loss_obj, x, y_true, seq_lengths, tolerance, h_init=None):
        assert x.ndim == 3
        num_input_params = x.shape[0] * x.shape[1] * x.shape[2]

        assert np.all(np.max(seq_lengths) == seq_lengths)

        # needed only for setting y_true (should be a separate call)
        loss_obj.forward_backwards(x, y_true, seq_lengths)

        print("Starting inputs parameters gradient check for: %s. Number parameters: %d"
              % (loss_obj.get_class_fq_name(), num_input_params))
        start_time = time.time()
        if h_init is None:
            success = gc.gradient_check(loss_obj.forward_backwards_grad_input, x, tolerance)
        else:
            success = gc.gradient_check(lambda: loss_obj.forward_backwards_grad_input(h_init=h_init), x, tolerance)
        self.assertTrue(success)
        time_elapsed = (time.time() - start_time)

        num_invocations = 1 + 2 * num_input_params
        print("per invocation time: %.4g sec" % (time_elapsed / num_invocations))
        print("total time elapsed for %d invocations: %.4g sec" % (num_invocations, time_elapsed))

        self.assertTrue(np.alltrue(np.equal(loss_obj.get_model(), loss_obj.get_built_model())))
        self.assertTrue(np.alltrue(np.equal(loss_obj.get_gradient(), loss_obj.get_built_gradient())))
