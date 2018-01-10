import unittest

import numpy as np

import gradient_check_test_shared as gcs
import pyneural.ce_crf_layer as ce_crf
import pyneural.neural_layer as nl
import pyneural.rnn_layer as rl
from pyneural.neural_base import LossNN


def create_random_data(loss_nn, num_samples, dim_h, dim_k):
    num_params = loss_nn.get_num_p()
    dtype = loss_nn.get_dtype()

    np.random.seed(seed=47)
    params = 0.1 * np.random.standard_normal(num_params).astype(dtype)
    data = np.random.standard_normal((num_samples, dim_h)).astype(dtype)

    labels = np.random.randint(0, dim_k, num_samples)
    last_label = np.random.randint(0, dim_k)

    return params, data, labels, last_label


class CRFandRRN(LossNN):
    def __init__(self, dim_d, dim_h, dim_k, max_batch_size, dtype):
        self.dim_d, self.dim_h, self.dim_k = dim_d, dim_h, dim_k
        self.ce_crf = ce_crf.CRFLayer(self.dim_k, dtype)
        self.nl = nl.NeuralLayer(self.dim_h, self.dim_k, dtype, activation=None)
        self.rnn = rl.RnnLayer(self.dim_d, self.dim_h, max_batch_size, dtype)
        num_p = self.ce_crf.get_num_p() + self.nl.get_num_p() + self.rnn.get_num_p()
        super(CRFandRRN, self).__init__(num_p, dtype)
        self.delta_error = None

    def get_display_dict(self):
        d = self._init_display_dict()
        d['ce_crf'] = self.ce_crf.get_display_dict()
        d['nl'] = self.nl.get_display_dict()
        d['rnn'] = self.rnn.get_display_dict()
        return d

    def set_prev_label(self, prev_label):
        self.ce_crf.set_prev_label(prev_label)

    def set_init_h(self, init_h):
        self.rnn.set_init_h(init_h)

    def forward_backwards(self, data, labels):
        self._x, self._y_true = data, labels

        hs = self.rnn.forward(data)
        hls = self.nl.forward(hs)
        loss, grad, delta_err_ce_crf = self.ce_crf.forward_backwards(hls, labels)
        delta_err_nl = self.nl.backwards(delta_err_ce_crf)
        self.delta_error = self.rnn.backwards(delta_err_nl)

        return loss, self.get_gradient(), self.delta_error

    def forward_backwards_grad_model(self, **kwargs):
        self.set_init_h(kwargs["h_init"])
        return super(CRFandRRN, self).forward_backwards_grad_model()

    def forward_backwards_grad_input(self, **kwargs):
        # set the (same) init, because after each forward propagation it is overwritten
        self.set_init_h(kwargs["h_init"])
        return super(CRFandRRN, self).forward_backwards_grad_input()

    def _set_model_references_in_place(self):
        params = self._model
        ofs1 = 0
        ofs2 = self.ce_crf.get_num_p()
        self.ce_crf.set_model_storage(params[ofs1:ofs2])
        ofs1 = ofs2
        ofs2 += self.nl.get_num_p()
        self.nl.set_model_storage(params[ofs1:ofs2])
        ofs1 = ofs2
        ofs2 += self.rnn.get_num_p()
        self.rnn.set_model_storage(params[ofs1:ofs2])

    def _set_gradient_references_in_place(self):
        grad = self._grad
        ofs1 = 0
        ofs2 = self.ce_crf.get_num_p()
        self.ce_crf.set_gradient_storage(grad[ofs1:ofs2])
        ofs1 = ofs2
        ofs2 += self.nl.get_num_p()
        self.nl.set_gradient_storage(grad[ofs1:ofs2])
        ofs1 = ofs2
        ofs2 += self.rnn.get_num_p()
        self.rnn.set_gradient_storage(grad[ofs1:ofs2])

    def get_built_model(self):
        return np.concatenate((self.ce_crf.get_model(), self.nl.get_model(), self.rnn.get_model()))

    def get_built_gradient(self):
        return np.concatenate((self.ce_crf.get_gradient(), self.nl.get_gradient(), self.rnn.get_gradient()))


class CRFLayerTest(gcs.GradientCheckTestShared):

    def test_trellis(self):
        dim_k = 3
        dtype, tolerance = np.float64, 1e-14
        # dtype, tolerance = np.float32, 1e-5

        model = np.array([[1.0, -2.0, 0.0], [-0.1, 1.0, 2.5], [-1.4, 1.1, 2.3]]).astype(dtype)
        self.assertEquals(model.shape, (dim_k, dim_k))
        params = np.reshape(model, (dim_k * dim_k,))

        last_label = 1
        labels = np.array([0, 1, 2, 1, 1])
        data = np.array([[1.5, -0.1, 0.1], [1.1, 2.5, -0.4], [0.5, -0.1, 5.0],
                         [-0.3, 1.54, 0.1], [0.1, 1.7, -0.1]]).astype(dtype)

        self.assertEquals(len(labels), len(data))
        num_samples = data.shape[0]

        crf_layer = ce_crf.CRFLayer(dim_k, dtype)
        crf_layer.init_parameters_storage(params)
        crf_layer.set_prev_label(last_label)

        crf_layer.compute_trellis(data)

        # forward and reverse paths outcomes should be equal with very high numerical precision
        forward_log_sum_exp = crf_layer.all_seq_log_sum_exp()
        reverse_log_sum_exp = crf_layer.all_seq_log_sum_exp_reverse()
        self.assertLessEqual(np.fabs(1 - forward_log_sum_exp / reverse_log_sum_exp), tolerance)

        score_correct = crf_layer.score_for_seq(labels)
        all_seq_sum = crf_layer.all_seq_log_sum_exp()
        prob = np.exp(score_correct) / np.exp(all_seq_sum)
        print("enum=%f all_seq_sum=%f loss=%f prob=%f"
              % (score_correct, all_seq_sum, crf_layer.loss_func(labels), prob))

        labels1 = np.empty(num_samples, dtype=np.int)
        # score1[k] is sum of exp(score) of all possible paths that end in label k.
        score1 = np.zeros(dim_k, dtype=dtype)
        en1 = 0.0
        for i5 in range(dim_k):
            labels1[4] = i5
            for i2 in range(dim_k):
                labels1[1] = i2
                for i3 in range(dim_k):
                    labels1[2] = i3
                    for i4 in range(dim_k):
                        labels1[3] = i4
                        for i1 in range(dim_k):
                            labels1[0] = i1
                            score1[i5] += np.exp(crf_layer.score_for_seq(labels1))
            en1 += score1[i5]

        # total score from recursion and explicit summation of all paths should be equal with high numerical precision
        self.assertLessEqual(np.fabs(1 - en1 / np.exp(all_seq_sum)), tolerance)

    def test_trellis_short(self):
        dim_k = 3
        dtype, tolerance = np.float64, 1e-15

        model = np.array([[1.0, -2.0, 0.0], [-0.1, 1.0, 2.5], [-1.4, 1.1, 2.3]]).astype(dtype)
        self.assertEquals(model.shape, (dim_k, dim_k))
        params = np.reshape(model, (dim_k * dim_k,))

        last_label = 1
        labels = np.array([0, 1, 2])
        data = np.array([[1.5, -0.1, 0.1], [1.1, 2.5, -0.4], [0.5, -0.1, 5.0]]).astype(dtype)

        self.assertEquals(len(labels), len(data))
        num_samples = data.shape[0]

        crf_layer = ce_crf.CRFLayer(dim_k, dtype)
        crf_layer.init_parameters_storage(params)
        crf_layer.set_prev_label(last_label)

        crf_layer.compute_trellis(data)

        # forward and reverse paths outcomes should be equal with very high numerical precision
        forward_log_sum_exp = crf_layer.all_seq_log_sum_exp()
        reverse_log_sum_exp = crf_layer.all_seq_log_sum_exp_reverse()
        self.assertLessEqual(np.fabs(1 - forward_log_sum_exp / reverse_log_sum_exp), tolerance)

        score_correct = crf_layer.score_for_seq(labels)
        all_seq_sum = crf_layer.all_seq_log_sum_exp()
        prob = np.exp(score_correct) / np.exp(all_seq_sum)
        print("enum=%f all_seq_sum=%f loss=%f prob=%f"
              % (score_correct, all_seq_sum, crf_layer.loss_func(labels), prob))

        labels1 = np.empty(num_samples, dtype=np.int)
        score1 = np.zeros(dim_k, dtype=dtype)
        en1 = 0.0
        for i3 in range(dim_k):
            labels1[2] = i3
            for i2 in range(dim_k):
                labels1[1] = i2
                for i1 in range(dim_k):
                    labels1[0] = i1
                    score1[i3] += np.exp(crf_layer.score_for_seq(labels1))
            en1 += score1[i3]

        # total score from recursion and explicit summation of all paths should be equal with high numerical precision
        self.assertLessEqual(np.fabs(1 - en1 / np.exp(all_seq_sum)), tolerance)

    def test_most_likely_sequence(self):
        num_samples = 4
        dim_k = 5
        dtype = np.float64

        crf_layer = ce_crf.CRFLayer(dim_k, dtype)

        model, data, _, last_label = create_random_data(crf_layer, num_samples, dim_k, dim_k)

        crf_layer.init_parameters_storage(model)
        crf_layer.set_prev_label(last_label)

        crf_layer.compute_trellis(data)

        labels1 = np.empty(num_samples, dtype=np.int)
        scores = np.empty((dim_k, dim_k, dim_k, dim_k), dtype=dtype)
        for i4 in range(dim_k):
            labels1[3] = i4
            for i3 in range(dim_k):
                labels1[2] = i3
                for i2 in range(dim_k):
                    labels1[1] = i2
                    for i1 in range(dim_k):
                        labels1[0] = i1
                        scores[i1, i2, i3, i4] = crf_layer.score_for_seq(labels1)

        sorted_scores = np.sort(np.reshape(scores, (dim_k**num_samples, )))
        # while argmax uses sum of all possible paths instead of only the (estimated) best path so far, the results
        # are surprisingly close
        most_probable_seq0 = np.argmax(crf_layer.s, axis=1)
        most_probable_seq = crf_layer.get_most_probable_seq()  # proper Viterbi
        # interestingly, in all executions with different parameters I tried, the sequence most_probable_seq0,
        # most_probable_seq differed only at the last label or were the same
        most_probable_seq_score0 = crf_layer.score_for_seq(most_probable_seq0)
        most_probable_seq_score = crf_layer.score_for_seq(most_probable_seq)
        print("scores=%f, %f" % (most_probable_seq_score0, most_probable_seq_score))
        self.assertLessEqual(most_probable_seq_score0, most_probable_seq_score)  # probably not guaranteed (?)

        pos = np.searchsorted(sorted_scores, most_probable_seq_score)
        ratio = float(pos) / float(sorted_scores.size)

        print("percentile=%f" % (float(pos) / float(sorted_scores.size)))
        self.assertLessEqual(0.98, ratio)

    @staticmethod
    def test_debug_enabled():
        num_samples = 20
        dim_k = 10
        dtype = np.float64

        crf_layer = ce_crf.CRFLayer(dim_k, dtype)

        model, data, labels, last_label = create_random_data(crf_layer, num_samples, dim_k, dim_k)

        crf_layer.init_parameters_storage(model)
        crf_layer.set_prev_label(last_label)

        crf_layer.compute_trellis_debug(data)

        crf_layer.set_labels(labels)
        crf_layer.backwards()
        crf_layer.post_backwards_assert()

    def test_layer_gradients(self):
        num_samples = 20
        dim_k = 10
        dtype, tolerance = np.float64, 1e-9

        crf_layer = ce_crf.CRFLayer(dim_k, dtype)

        model, data, labels, last_label = create_random_data(crf_layer, num_samples, dim_k, dim_k)

        crf_layer.init_parameters_storage(model)
        crf_layer.set_prev_label(last_label)

        self.do_param_gradient_check(crf_layer, data, labels, tolerance)
        self.do_input_gradient_check(crf_layer, data, labels, tolerance)

    def test_crf_layer_rnn_gradients(self):
        num_samples = 20
        dim_d, dim_h, dim_k = 8, 12, 5

        dtype, tolerance = np.float64, 1e-9

        loss_nn = CRFandRRN(dim_d, dim_h, dim_k, num_samples, dtype)

        model, data, labels, last_label = create_random_data(loss_nn, num_samples, dim_d, dim_k)
        h_init = 0.01 * np.random.standard_normal(dim_h).astype(dtype)

        loss_nn.init_parameters_storage(model=model)
        loss_nn.set_prev_label(last_label)

        self.do_param_gradient_check(loss_nn, data, labels, tolerance, h_init)
        self.do_input_gradient_check(loss_nn, data, labels, tolerance, h_init)


if __name__ == "__main__":
    unittest.main()
