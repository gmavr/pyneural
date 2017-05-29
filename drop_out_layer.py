import numpy as np


class DropoutLayer(object):
    """ Dropout layer. In-place modification of forward inputs and backwards delta error.
    
    At training time randomly zeroes each input with probability drop_probability.
    At test / evaluation time it does not drop any input but scales all inputs so that the average input is at the same
    scale as the average input with dropout at training time.
    
    The need to remember the drop_probability used during training and scale all inputs at evaluation time is a
    drawback that InvertedDropoutLayer was made to address.
    
    """
    def __init__(self, drop_probability):
        """
        Args:
            drop_probability: Set each input to 0 w.p. drop_probability otherwise leave unchanged
        """
        assert 0.0 <= drop_probability <= 1.0
        self._retain_probability = 1.0 - drop_probability
        self._drop_out_mask = None

    def get_display_dict(self):
        return {"retain_probability": self._retain_probability}
    
    def forward_train(self, x):
        """ Forward pass during training.
        
        Drops each coordinate randomly with probability (1 - retain_probability).
        Changes the supplied input x *in-place* and returns reference to it.
        If the lower layer retains a reference to its output, then its content will be changed.
        """
        self._drop_out_mask = np.random.binomial(1, self._retain_probability, size=x.shape)
        # do not do: .astype(self.dtype), will be done automatically if needed
        x *= self._drop_out_mask
        return x

    def forward_eval(self, x):
        """ Forward pass during evaluation / testing.
        
        Retains all input dimensions but scales them down by the retain_probability
        Changes the supplied input x *in-place* and returns reference to it.
        So if the lower layer retains a reference to its output, then its content will be changed.
        """
        x *= self._retain_probability
        return x

    def backwards(self, delta_err_in):
        """
        Changes the supplied error *in-place* and returns reference to it.
        So if the upper layer retains a reference to its delta_error, then its content will be changed.
        """
        delta_err_in *= self._drop_out_mask
        return delta_err_in

    def forward_train_with_fixed_mask(self, x):
        """
        Testing only, for gradient checks.
        Only difference from forward_train() is that it does not set a new random drop_out_mask.
        """
        x *= self._drop_out_mask
        return x

    def set_drop_out_mask(self, drop_out_mask):
        """
        Testing only, for gradient checks.
        """
        self._drop_out_mask = drop_out_mask

    def get_drop_out_mask(self):
        return self._drop_out_mask


class InvertedDropoutLayer(object):
    """ Inverted Dropout layer. No scaling needed at evaluation / test time.
    
    Unlike DropoutLayer, I could not get gradient_check to pass with updating x and delta_err in-place.
    This layer keeps at training its own copy or y and delta_err.
    """

    def __init__(self, drop_probability):
        """
         Args:
             drop_probability: Set each input to 0 w.p. drop_probability,
                otherwise multiply with 1 / (1 - drop_probability)
         """
        assert 0.0 <= drop_probability <= 1.0
        self._retain_probability = 1.0 - drop_probability
        self._scale_factor = 1.0 / self._retain_probability
        self._scaled_drop_out_mask = None
        self._y = None
        self.delta_err = None

    def get_display_dict(self):
        return {"retain_probability": self._retain_probability}

    def get_y(self):
        return self._y

    def forward_train(self, x):
        """ Forward pass during training.
        
        Drops each coordinate randomly with probability (1 - retain_probability) and scales it by 1 / retain_probability
        """
        self._scaled_drop_out_mask = np.random.binomial(1, self._retain_probability, size=x.shape).astype(x.dtype)
        self._scaled_drop_out_mask *= self._scale_factor

        # parallelism with vanilla neural layer:
        # (N, Dx) x (Dx, Dy) returns (N, Dy)
        # z = np.dot(self.x, self.w.T) + self.b
        # self.y = self.activation(z)
        # here: Dx = Dy, w = I_D_x,  self.b = 0
        # self.activation(z) = c * mult with binomial flag

        self._y = x * self._scaled_drop_out_mask
        return self._y

    def forward_eval(self, x):
        """ Forward pass during evaluation / testing.
        
        The whole purpose of InvertedDropoutLayer is for this operation to be a NO-OP
        """
        self._y = x
        return self._y

    def backwards(self, delta_err_in):
        # chain rule: delta_err_out = dj/dx = dj/dy * dy/dx = delta_err_in * self._scaled_drop_out_mask
        self.delta_err = delta_err_in * self._scaled_drop_out_mask
        return self.delta_err

    def forward_train_with_fixed_mask(self, x):
        """
        Testing only, for gradient checks.
        Only difference from forward_train() is that it does not set a new random drop_out_mask.
        """
        self._y = x * self._scaled_drop_out_mask
        return self._y

    def set_drop_out_mask(self, drop_out_mask):
        """
        Testing only, for gradient checks.
        """
        self._scaled_drop_out_mask = drop_out_mask * self._scale_factor

    def get_drop_out_mask(self):
        return self._scaled_drop_out_mask / self._scale_factor

    def get_scaled_drop_out_mask(self):
        return self._scaled_drop_out_mask
