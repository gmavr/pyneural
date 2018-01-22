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
    prev_label = np.random.randint(0, dim_k)

    return params, data, labels, prev_label


class CRFandRRN(LossNN):
    def __init__(self, dim_d, dim_h, dim_k, max_seq_length, dtype):
        self.dim_d, self.dim_h, self.dim_k = dim_d, dim_h, dim_k
        self.ce_crf = ce_crf.CRFLayer(self.dim_k, max_seq_length, dtype)
        self.nl = nl.NeuralLayer(self.dim_h, self.dim_k, dtype, activation=None)
        self.rnn = rl.RnnLayer(self.dim_d, self.dim_h, max_seq_length, dtype)
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
        dtype, rtol, atol = np.float64, 1e-14, 1e-15
        # dtype, rtol, atol = np.float32, 1e-6, 1e-15

        prev_label = 1
        seq = np.array([0, 1, 2, 1, 1], dtype=np.int32)
        data = np.array([[1.5, -0.1, 0.1], [1.1, 2.5, -0.4], [0.5, -0.1, 5.0],
                         [-0.3, 1.54, 0.1], [0.1, 1.7, -0.1]]).astype(dtype)
        self.assertEquals(len(seq), len(data))
        num_samples = data.shape[0]

        model = np.array([[1.0, -2.0, 0.0], [-0.1, 1.0, 2.5], [-1.4, 1.1, 2.3]], dtype=dtype)
        # model[prev_label].fill(0.0)  # zero the effect of previous state, for compatibility with tf.contrib.crf
        self.assertEquals(model.shape, (dim_k, dim_k))
        params = np.reshape(model, (dim_k * dim_k,))

        crf_layer = ce_crf.CRFLayer(dim_k, data.shape[0], dtype)
        crf_layer.init_parameters_storage(params)
        crf_layer.set_prev_label(prev_label)

        crf_layer.compute_trellis(data)

        # forward and reverse paths outcomes should be equal with very high numerical precision
        forward_log_sum_exp = crf_layer.all_seq_log_sum_exp()
        reverse_log_sum_exp = crf_layer.all_seq_log_sum_exp_reverse()
        self.assertTrue(np.allclose(forward_log_sum_exp, reverse_log_sum_exp, rtol=rtol, atol=atol))

        # verify that explicitly collecting the scores of all sequences (cost T^K) is the same as the recurrence
        # (dynamic programming) method (cost T*K^2) with high numerical precision

        seq1 = np.empty(num_samples, dtype=np.int)
        # scores[k] is sum of exp(score) of all possible paths that end in label k.
        scores = np.zeros(dim_k, dtype=dtype)
        all_seq_scores_sum_exp = 0.0
        for i5 in range(dim_k):
            seq1[4] = i5
            for i2 in range(dim_k):
                seq1[1] = i2
                for i3 in range(dim_k):
                    seq1[2] = i3
                    for i4 in range(dim_k):
                        seq1[3] = i4
                        for i1 in range(dim_k):
                            seq1[0] = i1
                            scores[i5] += np.exp(crf_layer.score_for_seq(seq1))
            all_seq_scores_sum_exp += scores[i5]

        all_seq_scores_log_sum = crf_layer.all_seq_log_sum_exp()

        # total score from recursion and explicit summation of all paths should be equal with high numerical precision
        self.assertTrue(np.allclose(all_seq_scores_sum_exp, np.exp(all_seq_scores_log_sum), rtol=rtol, atol=atol))

        seq_score = crf_layer.score_for_seq(seq)
        prob = np.exp(seq_score) / np.exp(all_seq_scores_log_sum)
        print("seq_score=%f all_seq_scores_log_sum=%f loss=%f prob=%f"
              % (seq_score, all_seq_scores_log_sum, crf_layer.loss_for_seq(seq), prob))

    def test_most_likely_sequence(self):
        num_samples = 4
        dim_k = 5
        dtype = np.float64

        crf_layer = ce_crf.CRFLayer(dim_k, num_samples, dtype)

        model, data, _, prev_label = create_random_data(crf_layer, num_samples, dim_k, dim_k)

        crf_layer.init_parameters_storage(model)
        crf_layer.set_prev_label(prev_label)

        crf_layer.compute_trellis(data)

        # most_probable_seq_greedy is the result of a simple but suboptimal greedy decoding strategy, which is 
        # surprisingly close to the optimal one returned by Viterbi decoding
        most_probable_seq_greedy = np.argmax(crf_layer.s, axis=1)
        most_probable_seq_score_greedy = crf_layer.score_for_seq(most_probable_seq_greedy)

        # proper Viterbi decoding
        most_probable_seq, most_probable_seq_score_viterbi = crf_layer.get_most_probable_seq()
        most_probable_seq_score = crf_layer.score_for_seq(most_probable_seq)
        self.assertTrue(np.allclose(most_probable_seq_score, most_probable_seq_score_viterbi, rtol=1e-15, atol=1e-15))

        self.assertLessEqual(most_probable_seq_score_greedy, most_probable_seq_score)

        # uncomment following for inspection
        #
        # seq = np.empty(num_samples, dtype=np.int)
        # scores = np.empty((dim_k, dim_k, dim_k, dim_k), dtype=dtype)
        # for i4 in range(dim_k):
        #     seq[3] = i4
        #     for i3 in range(dim_k):
        #         seq[2] = i3
        #         for i2 in range(dim_k):
        #             seq[1] = i2
        #             for i1 in range(dim_k):
        #                 seq[0] = i1
        #                 scores[i1, i2, i3, i4] = crf_layer.score_for_seq(seq)
        # sorted_scores = np.sort(np.reshape(scores, (dim_k**num_samples, )))
        # pos = np.searchsorted(sorted_scores, most_probable_seq_score_greedy)
        # print("greedy decoding score rank: %d/%d" % (pos + 1, sorted_scores.size))
        # print(most_probable_seq_greedy)
        # print(most_probable_seq)

    @staticmethod
    def test_debug_enabled():
        num_samples = 20
        dim_k = 10
        dtype = np.float64

        crf_layer = ce_crf.CRFLayer(dim_k, num_samples + 1, dtype)

        model, data, labels, prev_label = create_random_data(crf_layer, num_samples, dim_k, dim_k)

        crf_layer.init_parameters_storage(model)
        crf_layer.set_prev_label(prev_label)

        crf_layer.compute_trellis_debug(data)

        crf_layer.set_labels(labels)
        crf_layer.backwards()
        crf_layer.post_backwards_assert()

    def test_layer_gradients(self):
        num_samples = 20
        dim_k = 10
        dtype, tolerance = np.float64, 1e-9

        crf_layer = ce_crf.CRFLayer(dim_k, num_samples + 2, dtype)

        model, data, labels, prev_label = create_random_data(crf_layer, num_samples, dim_k, dim_k)

        crf_layer.init_parameters_storage(model)
        crf_layer.set_prev_label(prev_label)

        self.do_param_gradient_check(crf_layer, data, labels, tolerance)
        self.do_input_gradient_check(crf_layer, data, labels, tolerance)

    def test_crf_layer_rnn_gradients(self):
        num_samples = 20
        dim_d, dim_h, dim_k = 8, 12, 5

        dtype, tolerance = np.float64, 1e-9

        loss_nn = CRFandRRN(dim_d, dim_h, dim_k, num_samples, dtype)

        model, data, labels, prev_label = create_random_data(loss_nn, num_samples, dim_d, dim_k)
        h_init = 0.01 * np.random.standard_normal(dim_h).astype(dtype)

        loss_nn.init_parameters_storage(model=model)
        loss_nn.set_prev_label(prev_label)

        self.do_param_gradient_check(loss_nn, data, labels, tolerance, h_init)
        self.do_input_gradient_check(loss_nn, data, labels, tolerance, h_init)


if __name__ == "__main__":
    unittest.main()
