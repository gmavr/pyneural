import numpy as np

import pyneural.neural_base as nb


class LayerWithL2Loss(nb.LossNN):
    """ Adds a scalar (half of) squared error output loss function on top of an existing ComponentNN.

    It recognizes only the h_init optional argument and passes it appropriately.
    """

    def __init__(self, component_nn):
        assert isinstance(component_nn, nb.ComponentNN)
        self.component_nn = component_nn
        super(LayerWithL2Loss, self).__init__(self.component_nn.get_num_p(), component_nn.get_dtype())

    def get_display_dict(self):
        d = self._init_display_dict()
        d["component_nn"] = self.component_nn.get_display_dict()
        return d

    def _set_model_references_in_place(self):
        self.component_nn.set_model_storage(self._model)

    def _set_gradient_references_in_place(self):
        self.component_nn.set_gradient_storage(self._grad)

    def forward_backwards(self, x, y_true):
        self._x, self._y_true = x, y_true

        y = self.component_nn.forward(x)
        assert y_true.shape == y.shape

        # delta_err is the derivative of loss w.r. to the outputs of the lower layer
        delta_err = y - y_true
        loss = 0.5 * np.sum(delta_err * delta_err)
        delta_err2 = self.component_nn.backwards(delta_err)
        return loss, self._grad, delta_err2

    def forward_backwards_grad_model(self, **kwargs):
        # set the (same) init, because after each forward propagation it is overwritten
        if "h_init" in kwargs:
            self.component_nn.set_init_h(kwargs["h_init"])
        return super(LayerWithL2Loss, self).forward_backwards_grad_model()

    def forward_backwards_grad_input(self, **kwargs):
        # set the (same) init, because after each forward propagation it is overwritten
        if "h_init" in kwargs:
            self.component_nn.set_init_h(kwargs["h_init"])
        return super(LayerWithL2Loss, self).forward_backwards_grad_input()

    def get_built_gradient(self):
        return self.component_nn.get_gradient()

    def get_built_model(self):
        return self.component_nn.get_model()


class BatchSequencesWithL2Loss(nb.BatchSequencesLossNN):
    """ Adds a scalar (half of) squared error output loss function on top of an existing BatchSequencesLossNN.

    It recognizes only the h_init optional argument and passes it appropriately.
    """

    def __init__(self, component_nn, asserts_on=True):
        assert isinstance(component_nn, nb.BatchSequencesComponentNN)
        self.component_nn = component_nn
        self.asserts_on = asserts_on
        super(BatchSequencesWithL2Loss, self).__init__(
            num_p=self.component_nn.get_num_p(), dtype=component_nn.get_dtype())

    def get_display_dict(self):
        d = self._init_display_dict()
        d["component_nn"] = self.component_nn.get_display_dict()
        return d

    def _set_model_references_in_place(self):
        self.component_nn.set_model_storage(self._model)

    def _set_gradient_references_in_place(self):
        self.component_nn.set_gradient_storage(self._grad)

    def forward_backwards(self, x, y_true, seq_lengths):
        if self.asserts_on:
            # x is validated further in self.component_nn anyway
            assert seq_lengths.shape[0] == x.shape[1]
            assert np.issubdtype(seq_lengths.dtype, np.integer)
            assert seq_lengths.max() <= x.shape[0]

        num_sequences = x.shape[1]

        self._x, self._y_true, self._seq_lengths = x, y_true, seq_lengths

        y = self.component_nn.forward(x, seq_lengths)

        seq_length_dim_max = self.component_nn.get_max_seq_length_out()
        our_seq_lengths = self.component_nn.get_seq_lengths_out()

        if self.asserts_on:
            assert y_true.shape == y.shape
            assert y_true.dtype == y.dtype

        delta_err = y - y_true

        # There is no guarantee that elements in y_true after the sequence lengths are 0 and we MUST ignore them, so
        # zero out their contribution to error.
        if our_seq_lengths.min() != seq_length_dim_max:  # only 0-pad if necessary
            nb.zero_pad_overwrite(delta_err, seq_length_dim_max, num_sequences, our_seq_lengths)

        loss = 0.5 * np.sum(delta_err * delta_err)
        delta_err2 = self.component_nn.backwards(delta_err)
        return loss, self._grad, delta_err2

    def forward_backwards_grad_model(self, **kwargs):
        # set the (same) init, because after each forward propagation it is overwritten
        if "h_init" in kwargs:
            self.component_nn.set_init_h(kwargs["h_init"])
        return super(BatchSequencesWithL2Loss, self).forward_backwards_grad_model()

    def forward_backwards_grad_input(self, **kwargs):
        # set the (same) init, because after each forward propagation it is overwritten
        if "h_init" in kwargs:
            self.component_nn.set_init_h(kwargs["h_init"])
        return super(BatchSequencesWithL2Loss, self).forward_backwards_grad_input()

    def get_built_gradient(self):
        return self.component_nn.get_gradient()

    def get_built_model(self):
        return self.component_nn.get_model()
