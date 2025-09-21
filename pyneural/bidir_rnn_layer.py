import numpy as np

import pyneural.gru_layer as gru
import pyneural.rnn_layer as rl
from pyneural.neural_base import ComponentNN


class BidirRnnLayer(ComponentNN):
    def __init__(self, dim_d, dim_h, max_seq_length, bptt_steps=None, dtype=np.float32, activation="tanh",
                 cell_type="basic", grad_clip_thres=None, asserts_on=True):
        """ Bidirectional Recurrent Neural Layer

        Args:
            dim_d: input dimension
            dim_h: hidden state dimension of each of the 2 encapsulated RNNs
            max_seq_length: maximum sequence length for any sequence that will be presented in forward and backwards.
            dtype: numpy type of all parameters and inputs: np.float32 or np.float64
            activation: activation (transfer) function string name. Ignored for cell_type different than "basic".
            cell_type: type of the 2 encapsulated RNNs. Valid values are "basic" and "gru".
            bptt_steps: length of truncated backpropagation. See RnnLayer for effect.
                Ignored for cell_type different than "basic".
            grad_clip_thres: See RnnLayer for effect. Ignored for cell_type different than "basic".
            asserts_on: perform invariant and consistency assertions. Recommended to set to False only at final steps of
                training large models
        """
        self.dim_d, self.dim_h = dim_d, dim_h
        self.asserts_on = asserts_on
        if bptt_steps is None:
            bptt_steps = max_seq_length
        if cell_type == "basic":
            self.rnn_f = rl.RnnLayer(self.dim_d, self.dim_h, max_seq_length, dtype, activation=activation,
                                     bptt_steps=bptt_steps, grad_clip_thres=grad_clip_thres, asserts_on=asserts_on)
            self.rnn_b = rl.RnnLayer(self.dim_d, self.dim_h, max_seq_length, dtype, activation=activation,
                                     bptt_steps=bptt_steps, grad_clip_thres=grad_clip_thres, asserts_on=asserts_on)
        elif cell_type == "gru":
            self.rnn_f = gru.GruLayer(self.dim_d, self.dim_h, max_seq_length, dtype=dtype, asserts_on=asserts_on)
            self.rnn_b = gru.GruLayer(self.dim_d, self.dim_h, max_seq_length, dtype=dtype, asserts_on=asserts_on)
        else:
            raise ValueError("Invalid cell type")
        super().__init__(self.rnn_f.get_num_p() + self.rnn_b.get_num_p(), dtype)
        self.delta_error = np.empty((max_seq_length, self.dim_d), dtype=dtype)

    def get_display_dict(self):
        d = self._init_display_dict()
        d['rnn_f'] = self.rnn_f.get_display_dict()
        d['rnn_b'] = self.rnn_b.get_display_dict()
        return d

    def model_normal_init(self, sd):
        self.rnn_f.model_normal_init(sd)
        self.rnn_b.model_normal_init(sd)

    def model_glorot_init(self):
        self.rnn_f.model_glorot_init()
        self.rnn_b.model_glorot_init()

    def model_identity_glorot_init(self):
        if isinstance(self.rnn_f, gru.GruLayer):
            raise ValueError("This initialization not supported for GruLayer")
        self.rnn_f.model_identity_glorot_init()
        self.rnn_b.model_identity_glorot_init()

    def get_max_seq_length(self):
        return self.rnn_f.get_max_seq_length()

    def set_init_h(self, h_init_f):
        """
        The reverse sequence initial state is always set to 0.0 before forward propagation and cannot be set by user.
        """
        self.rnn_f.set_init_h(h_init_f)
        self.rnn_b.reset_last_hidden()

    def reset_last_hidden(self):
        self.rnn_f.reset_last_hidden()
        self.rnn_b.reset_last_hidden()

    def forward(self, x):
        hs_f = self.rnn_f.forward(x)

        # by design always zero reverse sequence previous state
        self.rnn_b.reset_last_hidden()

        # reverse time dimension
        reversed_data = x[::-1]  # reverse view (no mem copy)
        hs_b = self.rnn_b.forward(reversed_data)
        reversed_hs_b = hs_b[::-1]  # reverse view (no mem copy)

        hs = np.concatenate((hs_f, reversed_hs_b), axis=1)  # has to copy memory unfortunately
        # assert hs.shape == (num_samples, 2 * self.dim_h)
        return hs

    def backwards(self, delta_err_in):
        num_samples = self.rnn_f.x.shape[0]
        if self.asserts_on:
            assert delta_err_in.shape == (num_samples, 2 * self.dim_h)

        # first half of matrix columns is projection for forward rnn
        delta_err_f = self.rnn_f.backwards(delta_err_in[:, 0:self.dim_h])

        # reverse rows (corresponding to time) and take second half of matrix columns
        delta_err_2_reversed = delta_err_in[::-1, self.dim_h:(2*self.dim_h)]
        delta_err_b = self.rnn_b.backwards(delta_err_2_reversed)
        delta_err_b_reversed = delta_err_b[::-1]

        np.add(delta_err_f, delta_err_b_reversed, out=self.delta_error[0:num_samples])
        return self.delta_error[0:num_samples]

    def _set_model_references_in_place(self):
        params = self._model
        self.rnn_f.set_model_storage(params[0:self.rnn_f.get_num_p()])
        self.rnn_b.set_model_storage(params[self.rnn_f.get_num_p():self._num_p])

    def _set_gradient_references_in_place(self):
        grad = self._grad
        self.rnn_f.set_gradient_storage(grad[0:self.rnn_f.get_num_p()])
        self.rnn_b.set_gradient_storage(grad[self.rnn_f.get_num_p():self._num_p])

    def get_built_model(self):
        return np.concatenate((self.rnn_f.get_model(), self.rnn_b.get_model()))

    def get_built_gradient(self):
        return np.concatenate((self.rnn_f.get_gradient(), self.rnn_b.get_gradient()))
