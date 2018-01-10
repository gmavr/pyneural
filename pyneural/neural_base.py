from abc import ABCMeta, abstractmethod

import numpy as np


class CoreNN(object):
    __metaclass__ = ABCMeta

    def __init__(self, num_p, dtype):
        """
        Args:
            num_p: Number of parameters in the model (dimension of the model parameter vector)
            dtype: numpy float type: valid values np.float32, np.float64
        """
        if num_p <= 0:
            raise ValueError("num_p must be positive integer, supplied %s" % str(num_p))
        if dtype != np.float32 and dtype != np.float64:
            raise ValueError("dtype must be np.float32 or np.float64, supplied %s" % str(dtype))
        self._num_p = int(num_p)
        self._dtype = dtype
        self._model = None
        self._grad = None

    def get_num_p(self):
        """
        Return:
            Number of parameters in the model (dimension of the model parameter vector)
        """
        return self._num_p

    def get_dtype(self):
        return self._dtype

    def init_parameters_storage(self, model=None, grad=None):
        """ Use provided storage for model and model gradient or allocate storage.
        
        IT IS REQUIRED this method or both of set_model_storage(), set_gradient_storage() are invoked manually before
        the object can be used.
        (This method is necessary because for compound objects at __init()__ often we can't know the size of the model
        and gradient for their full arrays to be allocated as contiguous memory blocks).
        In all cases, after method returns successfully internal references for the model and the gradient have been
        set up. If any of the model or grad are passed as non-None then references to these (and inside them) are
        created. If passed as None, then new internal memory is allocated and references set the same way. Memory is not
        initialized to any particular values.
        
        Args:
            model: If not None, model np.array of shape (self.get_num_p(), ) to be used for the lifetime of the object
                until init_parameters_storage() or set_model() is invoked.
            grad: If not None, gradient np.array of shape (self.get_num_p(), ) to be used for the lifetime of the object
                until init_parameters_storage() or set_gradient_storage() is invoked.

        """
        if model is None:
            self._model = np.empty((self._num_p,), dtype=self._dtype)
        else:
            assert model.shape == (self._num_p,) and model.dtype == self._dtype
            self._model = model
        self._set_model_references_in_place()

        if grad is None:
            self._grad = np.empty((self._num_p,), dtype=self._dtype)
        else:
            assert grad.shape == (self._num_p,) and grad.dtype == self._dtype
            self._grad = grad
        self._set_gradient_references_in_place()

    def set_model_storage(self, model):
        """ Set model buffer and recursively internal references within that buffer.
        
        Sets the memory storage where the model gradient is kept and creates internal references to appropriate
        locations within that memory buffer.
        By design it does not copy passed vector.
        The assumption is that a single continuous memory block has been allocated before outside of this class
        and when wiring components internal references are set inside it.
        It is NOT allowed to replace existing model storage.
        
        Args:
            model: current model parameters, np.array of shape (self.get_num_p(), )
        Raises:
            AssertionError: on consistency errors (todo: better to raise ValueError)
        """
        assert model.shape == (self._num_p,)
        assert model.dtype == self._dtype
        assert self._model is None
        self._model = model
        self._set_model_references_in_place()

    def set_gradient_storage(self, grad):
        """ Set model gradient buffer and recursively internal references within that buffer.
        
        Sets the memory storage where the model gradient is kept and creates internal references to appropriate
        locations within that memory buffer.
        By design it does not copy passed vector.
        The assumption is that a single continuous memory block has been allocated before outside of this class
        and when wiring components internal references are set inside it.
        It is NOT allowed to overwrite existing gradient storage.
        
        Args:
            grad: storage for the gradient of the model parameters, np.array of shape (self.get_num_p(), )
        Raises:
            AssertionError: on consistency errors (todo: better to raise ValueError)
        """
        assert grad.shape == (self._num_p,)
        assert grad.dtype == self._dtype
        assert self._grad is None
        self._grad = grad
        self._set_gradient_references_in_place()

    @abstractmethod
    def _set_model_references_in_place(self):
        """ Sets internal matrices to the appropriate locations inside the the model vector.
        
        Returns:
            None
        """

    @abstractmethod
    def _set_gradient_references_in_place(self):
        """
        Sets internal matrices to the appropriate locations inside the gradient vector of the model.
        
        When necessary, it is your responsibility to set gradient to 0 (suggested way is to cal np.fill() on the 
        terminal nodes of the graph.)
        
        Returns:
            None
        """

    def get_model(self):
        """ Returns reference to the internal one-dimensional vector of model.

        Returns:
            model: numpy.array of shape (get_num_p(), )
        """
        return self._model

    def get_gradient(self):
        """ Returns reference to the internal one-dimensional vector of model gradient.
        
        Returns:
            gradient: gradient wrt the current parameters of the delta_err_in fed in the last invocation of
                back_propagation_batch(); numpy.array of shape (get_num_p(), )
        """
        return self._grad

    @abstractmethod
    def get_built_model(self):
        """ Assemble model on a new memory buffer different than the internal one. Test only.
         
        Returns the same contents as get_model() but not necessarily backed by the same memory.
        
        Returns:
            model: numpy.array of shape (get_num_p(), )
        """

    @abstractmethod
    def get_built_gradient(self):
        """ Assemble model gradient on a new memory buffer different than the internal one. Test only. 
        
        Returns the same contents as get_gradient() but not necessarily backed by the same memory.
        
        Returns:
            gradient: gradient wrt the current parameters of the delta_err_in fed in the last invocation of
                back_propagation_batch(); numpy.array of shape (get_num_p(), )
        """

    def get_class_fq_name(self):
        return self.__module__ + "." + self.__class__.__name__

    def _init_display_dict(self):
        return {"fq_name": self.get_class_fq_name(), "num_p": self.get_num_p()}

    @abstractmethod
    def get_display_dict(self):
        """
        Returns:
            Dictionary of selected class elements for human inspection.
            The main use case is to store human readable information (in json) about the classes comprising the model
            alongside the serialized model parameters.
        """


class ComponentNN(CoreNN):

    def __init__(self, num_p, dtype):
        super(ComponentNN, self).__init__(num_p, dtype)
        self.y = None  # leave this non-encapsulated

    @abstractmethod
    def forward(self, x):
        """
        Performs one batch forward propagation using the current model.
        Retains reference to passed input data matrix and assumes that it will NOT be changed externally until the next
        invocation of forward_batch()
        Args:
            x: N input observations, numpy.array of shape (N, input_dimension)
        Returns:
            y: layer forward output, numpy.array, usually of shape (N, output_dimension) or (output_dimension, )
        """

    @abstractmethod
    def backwards(self, delta_err_in):
        """
        Args:
            delta_err_in: error vector returned from upper layer, i.e. derivative of loss function w.r. to outputs
                of this layer; numpy.array of shape (N, output_dimension)  (or (output_dimension, ))
        Returns:
            delta_err_out: error vector from this layer, i.e. derivative of loss function w.r. to inputs to this layer
                numpy.array of shape (N, input_dimension)
        """


class LossNN(CoreNN):
    """
    Encapsulates one or more NN layers where the output of the top layer is a scalar loss function
    """

    def __init__(self, num_p, dtype):
        super(LossNN, self).__init__(num_p, dtype)
        self._x = None  # needed for gradient checks only really
        self._y_true = None  # needed for gradient checks only really

    @abstractmethod
    def forward_backwards(self, x, y_true):
        """ Performs one forward propagation and one back propagation using the current model.

        Args:
            x: N input observations, numpy.array of shape (N, input_dimension)
            y_true: target (true) values, numpy.array, usually of shape (N, )
                y_true is the (integer) class index for classification, a continuous value for regression
        Returns:
            loss: total loss over the N observations, scalar
            model_gradient: gradient of the loss wrt the current parameters,
                numpy.array of shape (parameter_dimension, )
            input_gradient:: gradient of the loss wrt the current inputs to this layer,
                numpy.array of shape (N, input_dimension)
        """

    def forward_backwards_grad_model(self, **kwargs):
        """ Convenience wrapper method for gradient_check.
        
        Use case: model changed in-place across invocations of this method in gradient check.
        Most common implementation included, but meant to be overridden as needed.
        
        Args:
            kwargs: optional model-specific arguments (e.g. initializations)
        Returns:
            loss: see forward_backwards()
            gradient: see forward_backwards()
        """
        loss, grad, _ = self.forward_backwards(self._x, self._y_true)
        return loss, grad

    def forward_backwards_grad_input(self, **kwargs):
        """ Convenience wrapper method for gradient_check.
        
        Use case: input x changed in-place across invocations of this method in gradient check/
        Most common implementation included, but meant to be overridden as needed.
        
        Returns:
            loss: see forward_backwards()
            delta_error: see forward_backwards()
        """
        loss, _, delta_err = self.forward_backwards(self._x, self._y_true)
        return loss, delta_err


class BatchSequencesComponentNN(CoreNN):

    def __init__(self, num_p, max_seq_length, batch_size, dtype):
        """
        Args:
            num_p: model dimensionality
            max_seq_length: maximum possible length of a sequence in any batch, i.e. upper limit on x.shape[0]
            batch_size: number of sequences in a batch, i.e. x.shape[1]
            dtype: floating arithmetic type (np.float32 or np.float64)
        """
        super(BatchSequencesComponentNN, self).__init__(num_p, dtype)
        self._max_seq_length = max_seq_length
        self._max_num_sequences = batch_size
        # self._curr_batch_seq_dim_length == x.shape[0] of last batch (last x passed in forward_batch(self))
        # It is allowed that no sequence in the batch has that length
        self._curr_batch_seq_dim_length = 0
        self._curr_num_sequences = 0
        self._curr_min_seq_length = 0
        self._curr_max_seq_length = 0
        self._seq_lengths = None

    @abstractmethod
    def forward(self, x, seq_lengths):
        """ Performs one batch forward propagation using the current model.
        
        Retains reference to passed input data matrix and assumes that it will NOT be changed externally until the next
        invocation of forward_batch()
        Sequences in the batch that are shorter than maximum sequence length M should be passed as 0-padded to length M.
        (but it seems that if padded elements are not set to 0s, their values will be ignored anyway?
        We still check that they are in fact 0.0 as an assertion on the lower layer)
        It is allowed that all sequences in the batch are shorter than M.
        It is allowed that one or more sequences have 0 length.
        The size M of first dimension of x must be less or equal to max_seq_length  chosen at object construction time.
        The number of sequences N passed must be equal to batch_size chosen at object construction time.
        
        Args:
            x: N input sequences, each at most M observations, numpy.array of shape (M, N, input_dimension)
            seq_lengths: the lengths of the N input sequences, numpy.array of shape (N, )
        Returns:
            y: layer forward output, numpy.array, usually of shape (M, N, output_dimension) (or (N, output_dimension))
        """

    @abstractmethod
    def backwards(self, delta_upper):
        """ Sets model parameters keeping internal hidden state unchanged
        
        Args:
            delta_upper: error vector returned from upper layer, i.e. derivative of loss function w.r. to outputs
                of this layer. numpy.array of shape (M, N, output_dimension)   (or (N, output_dimension))
                it must hold delta_upper[i, j, :] == 0 for any i, j with i > _seq_lengths[j]
        Returns:
            delta_err_out: error vector from this layer, i.e. derivative of loss function w.r. to inputs to this layer
                numpy.array of shape (M, N, input_dimension)
        """

    def get_max_seq_length(self):
        return self._max_seq_length

    def get_max_batch_size(self):
        return self._max_num_sequences

    def validate_zero_padding(self, array3d):
        """ Verifies that it holds array3d[i, j, :] == 0 for any i, j with i > self._seq_lengths[j]
        
        Args:
            array3d: numpy.array of shape (M, N, D)
        Raises:
            AssertionError: when checked invariant does not hold
        """
        if self._curr_min_seq_length == self._curr_batch_seq_dim_length:
            # optimization, nothing to check, return now
            return
        for j in xrange(self._curr_num_sequences):
            if self._seq_lengths[j] < self._curr_batch_seq_dim_length:
                assert np.alltrue(np.equal(array3d[self._seq_lengths[j]:self._curr_batch_seq_dim_length, j], 0.0))

    def zero_pad_overwrite(self, array3d):
        """ Sets array3d[i, j, :] == 0 for any i, j with i > self._seq_lengths[j]
        
        Args:
            array3d: numpy.array of shape (M, N, D)
        """
        if self._curr_min_seq_length == self._curr_batch_seq_dim_length:
            # optimization, nothing to do, return now
            return
        for j in xrange(self._curr_num_sequences):
            if self._seq_lengths[j] < self._curr_batch_seq_dim_length:
                array3d[self._seq_lengths[j]:self._curr_batch_seq_dim_length, j] = 0.0


class BatchSequencesLossNN(CoreNN):

    def __init__(self, num_p, max_seq_length, batch_size, dtype):
        super(BatchSequencesLossNN, self).__init__(num_p, dtype)
        self._max_seq_length = max_seq_length
        self._max_num_sequences = batch_size
        # self._curr_batch_seq_dim_length == x.shape[0] of current batch.
        # It is allowed that no sequence in the batch has that length
        self._curr_batch_seq_dim_length = 0
        self._curr_num_sequences = 0
        self._curr_min_seq_length = 0   # minimum sequence length in the current batch
        self._curr_max_seq_length = 0   # maximum sequence length in the current batch
        self._seq_lengths = None
        self._x = None  # needed for gradient checks only really
        self._y_true = None  # needed for gradient checks only really

    @abstractmethod
    def forward_backwards(self, x, y_true, seq_lengths):
        """ Performs one forward propagation and one back propagation using the current model.

        Args:
            x: N input sequences, each with M observations, numpy.array of shape (M, N, input_dimension)
            seq_lengths: the lengths of each of N input sequences, numpy.array of shape (N, )
            y_true: target (true) values, numpy.array
                usually of shape (M, N, ), but can be (N, output_dimension) or (output_dimension, )
        Returns:
            loss: total loss over all valid observations, scalar
            gradient: gradient of the loss wrt the current parameters, numpy.array of shape (parameter_dimension, )
            delta_error: error vector from this layer, i.e. derivative w.r.to the inputs to this layer,
                None if not-differentiable, numpy.array of shape (M, N, input_dimension)
        """

    def forward_backwards_grad_model(self, **kwargs):
        """ Convenience wrapper method for gradient_check.
        
        Use case: model changed in-place across invocations of this method in gradient check.
        Most common implementation supplied as default, but meant to be overridden as needed.
        
        Args:
            kwargs: optional model-specific arguments (e.g. initializations)
        Returns:
            loss: see forward_backwards()
            gradient: see forward_backwards()
        """
        loss, grad, _ = self.forward_backwards(self._x, self._y_true, self._seq_lengths)
        return loss, grad

    def forward_backwards_grad_input(self, **kwargs):
        """ Convenience wrapper method for gradient_check.
        
        Most common implementation supplied as default, but meant to be overridden as needed.
        Returns:
            loss: see forward_backwards()
            delta_error: see forward_backwards()
        """
        loss, _, delta_err = self.forward_backwards(self._x, self._y_true, self._seq_lengths)
        return loss, delta_err

    def get_max_seq_length(self):
        return self._max_seq_length

    def get_max_batch_size(self):
        return self._max_num_sequences

    def validate_zero_padding(self, array3d):
        """ Verifies that it holds array3d[i, j, :] == 0 for any i, j with i > self._seq_lengths[j]
        
        Args:
            array3d: numpy.array of shape (M, N, D)
        Raises:
            AssertionError: when checked invariant does not hold
        """
        if self._curr_min_seq_length == self._curr_batch_seq_dim_length:
            # optimization, nothing to check, return now
            return
        for j in xrange(self._curr_num_sequences):
            if self._seq_lengths[j] < self._curr_batch_seq_dim_length:
                assert np.alltrue(np.equal(array3d[self._seq_lengths[j]:self._curr_batch_seq_dim_length, j], 0.0))


def glorot_init(shape):
    limit = np.sqrt(6.0 / (np.sum(shape)))
    return np.random.uniform(low=-limit, high=limit, size=shape)

