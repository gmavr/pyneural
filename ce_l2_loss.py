import numpy as np

import neural_base as nb


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

    def __init__(self, component_nn):
        assert isinstance(component_nn, nb.BatchSequencesComponentNN)
        self.component_nn = component_nn
        self.asserts_on = True
        super(BatchSequencesWithL2Loss, self).__init__(
            num_p=self.component_nn.get_num_p(), max_seq_length=component_nn.get_max_seq_length(),
            batch_size=component_nn.get_max_batch_size(), dtype=component_nn.get_dtype())

    def get_display_dict(self):
        d = self._init_display_dict()
        d["component_nn"] = self.component_nn.get_display_dict()
        return d

    def _set_model_references_in_place(self):
        self.component_nn.set_model_storage(self._model)

    def _set_gradient_references_in_place(self):
        self.component_nn.set_gradient_storage(self._grad)

    def forward_backwards(self, x, y_true, seq_lengths):
        # x is validated by self.component_nn anyway, do not duplicate here

        if self.asserts_on:
            assert seq_lengths.shape[0] == x.shape[1]
            assert seq_lengths.dtype == np.int
            assert np.max(seq_lengths) <= x.shape[0]

        self._curr_num_sequences = x.shape[1]
        self._curr_batch_seq_dim_length = x.shape[0]

        self._x, self._y_true, self._seq_lengths = x, y_true, seq_lengths

        # This object does not validate that elements past end-of-sequence of passed y_true are 0. But if y_true
        # contains non-zero for past-end-of-sequence elements, then the delta_err computed here will have non-zero
        # elements at locations where the component back propagation expects 0 and the error will be caught there.

        y = self.component_nn.forward(x, seq_lengths)

        if self.asserts_on:
            assert y_true.shape == y.shape

        delta_err = y - y_true
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
