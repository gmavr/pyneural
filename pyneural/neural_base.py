from abc import ABCMeta, abstractmethod

import numpy as np

import misc_cy


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
            if model.shape != (self._num_p,) or model.dtype != self._dtype:
                raise ValueError("Passed illegal model structure")
            self._model = model
        self._set_model_references_in_place()

        if grad is None:
            self._grad = np.empty((self._num_p,), dtype=self._dtype)
        else:
            assert grad.shape == (self._num_p,) and grad.dtype == self._dtype
            if grad.shape != (self._num_p,) or grad.dtype != self._dtype:
                raise ValueError("Passed illegal gradient structure")
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
            ValueError: on illegal arguments or invocation
        """
        if model.shape != (self._num_p,) or model.dtype != self._dtype:
            raise ValueError("Passed illegal buffer structure")
        if self._model is not None:
            raise ValueError("Model buffer already set")
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
            ValueError: on illegal arguments or invocation
        """
        if grad.shape != (self._num_p,) or grad.dtype != self._dtype:
            raise ValueError("Passed illegal buffer structure")
        if self._grad is not None:
            raise ValueError("Gradient buffer already set")
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
        # self._x and self._y_true  are needed for gradient checks
        self._x = None
        self._y_true = None

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
    """Layer taking as input a batch of sequences, where each sequence can have a different length.

    First dimension is time, second is sequence index.
    """

    def __init__(self, num_p, max_seq_length, max_batch_size, dtype):
        """
        Args:
            num_p: model dimensionality
            max_seq_length: maximum possible length of a sequence in any batch, i.e. upper limit on x.shape[0]
            max_batch_size: maximum possible number of sequences in any batch, i.e. upper limit on x.shape[1]
            dtype: floating arithmetic type (np.float32 or np.float64)
        """
        super(BatchSequencesComponentNN, self).__init__(num_p, dtype)
        self._max_seq_length = max_seq_length  # maximum length of dimension 0 (time) of x for all batches
        self._max_num_sequences = max_batch_size  # maximum length of dimension 1 (sequence) of x for all batches
        if self._max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive integer, supplied %s" % str(max_seq_length))
        if self._max_num_sequences <= 0:
            raise ValueError("max_batch_size must be positive integer, supplied %s" % str(max_batch_size))
        # minimum and max lengths of valid values in dimension 0 (time) input x
        self._curr_min_seq_length = 0
        self._curr_max_seq_length = 0
        self._curr_seq_length_dim_max = 0  # current input x.shape[0]
        # It is allowed that no sequence in the batch has length equal to the maximum dimension, i.e. it is allowed
        # that _curr_max_seq_length < _curr_seq_length_dim_max
        self._curr_num_sequences = 0  # current input x.shape[1]
        self._seq_lengths = None  # 1-d np.array

    @abstractmethod
    def forward(self, x, seq_lengths):
        """Performs one forward propagation using the current model.
        
        Retains reference to passed input data matrix and assumes that it will NOT be changed externally until the next
        invocation of this method.
        The values of the inputs after the end of each sequence length are required to be all 0.0 or 0. This method may
        rely on that.
        The values of the outputs after the end of each sequence length are produced as all 0.0 or 0.
        (This design decision regarding 0-padding is because all implementations in practice internally require them to
        be 0 and if not they would both have to check if 0 and set to 0.)
        It is allowed that all sequences in the batch are shorter than maxT.
        It is allowed that one or more sequences have 0 length.
        Size maxT of first dimension of x must be less or equal to max_seq_length chosen at object construction time.
        The number of sequences N passed must be equal to batch_size chosen at object construction time.
        
        Args:
            x: numpy.array of shape (maxT, N, ..)
                N input sequences, each at most maxT observations, 0-padded
            seq_lengths: numpy.array of np.integer of shape (N, )
                the lengths of the N input sequences
        Returns:
            y: layer forward output, numpy.array, usually of shape (maxT, N, output_dimension), 0-padded
        """

    @abstractmethod
    def backwards(self, delta_upper):
        """Performs one forward propagation using the current model (keeping internal model unchanged).

        The values of the input delta_upper after the end of each sequence length are required to be all 0.0 and this
        method can rely on that.
        The values of the delta_err_out after the end of each sequence length are produced as all 0.0.

        Args:
            delta_upper: error vector returned from upper layer, i.e. derivative of loss function w.r. to outputs
                of this layer. numpy.array of shape (maxT, N, output_dimension)   (or (N, output_dimension))
                it must hold delta_upper[i, j, :] == 0 for any i, j with i > _seq_lengths[j]
        Returns:
            delta_err_out: error vector from this layer, i.e. derivative of loss function w.r. to inputs to this layer
                numpy.array of shape (maxT, N, input_dimension), 0-padded
        """

    def get_max_seq_length(self):
        return self._max_seq_length

    def get_max_seq_length_out(self):
        # The common case as default implementation. Override as needed.
        return self._max_seq_length

    def get_max_batch_size(self):
        return self._max_num_sequences

    def get_seq_lengths(self):
        return self._seq_lengths

    def get_seq_lengths_out(self):
        """Return the output sequence lengths.
        """
        # The common case as default implementation. Override as needed.
        return self._seq_lengths

    def _set_lengths(self, x, seq_lengths):
        assert x.shape[1] == seq_lengths.shape[0]  # matrix must be provided trimmed
        self._curr_num_sequences = x.shape[1]
        self._curr_seq_length_dim_max = x.shape[0]
        self._seq_lengths = seq_lengths
        self._curr_min_seq_length = self._seq_lengths.min()
        self._curr_max_seq_length = self._seq_lengths.max()

    def _validate_zero_padding(self, array, max_seq_length=None):
        """Validates that array[i, j] == 0 for any i, j with seq_lengths[j] < i
        
        Args:
            array: numpy.array of shape (M, N, ...)
            max_seq_length: integer, maximum time to check up to
                if None, then self._curr_seq_length_dim_max
        Raises:
            AssertionError: when checked invariant does not hold
        """
        if max_seq_length is None:
            max_seq_length = self._curr_seq_length_dim_max
        if self._curr_min_seq_length == max_seq_length:
            # optimization, nothing to check, return now
            return
        validate_zero_padding(array, max_seq_length, self._curr_num_sequences, self._seq_lengths)

    def _zero_pad_overwrite(self, array):
        """Sets array[i, j] == 0 for any i, j with self._seq_lengths[j] < i
        
        Args:
            array: numpy.array of shape (M, N, ...)
        """
        if self._curr_min_seq_length == self._curr_seq_length_dim_max:
            # optimization, nothing to do, return now
            return
        zero_pad_overwrite(array, self._curr_seq_length_dim_max, self._curr_num_sequences, self._seq_lengths)


def validate_zero_padding(array_nd, max_seq_length, max_num_sequences, seq_lengths):
    """Validates that array_nd[i, j] == 0 for any i, j with seq_lengths[j] < i

    Note: depending on parameters, some elements at the end are not checked for 0-padding.

    Args:
        array_nd: numpy.array of shape (M, N) or (M, N, K)
        max_seq_length: maximum in dimension 0, <= M, checked up to max_seq_length, the rest are ignored
        max_num_sequences: maximum in dimension 1, <=N, the rest are ignored
        seq_lengths: 1-d array of lengths, elements after the lengths are checked for 0-padding
    Raises:
        AssertionError: when checked invariant does not hold
    """
    if array_nd.ndim == 2:
        assert misc_cy.validate_zero_padding_2d_int(array_nd, max_seq_length, max_num_sequences, seq_lengths)
    else:
        assert misc_cy.validate_zero_padding_3d(array_nd, max_seq_length, max_num_sequences, seq_lengths)
    # above is a faster equivalent of following:
    # for j in xrange(max_num_sequences):
    #     if seq_lengths[j] < max_seq_length:
    #         assert np.all(array_nd[seq_lengths[j]:max_seq_length, j] == 0.0)


def zero_pad_overwrite(array_nd, max_seq_length, max_num_sequences, seq_lengths):
    """Sets array_nd[i, j] == 0 for any i, j with seq_lengths[j] < i

    Note: depending on parameters, some elements at the end remain non-0-padded.

    Args:
        array_nd: numpy.array of shape (M, N) or in general (M, N, ...)
        max_seq_length: maximum in dimension 0, <= M, 0-padded up to max_seq_length, the rest are ignored
        max_num_sequences: maximum in dimension 1, <=N, the rest are ignored
        seq_lengths: 1-d array of lengths, this function 0-pads elements after the lengths up to max_seq_length
    """
    for j in xrange(max_num_sequences):
        if seq_lengths[j] < max_seq_length:
            array_nd[seq_lengths[j]:max_seq_length, j].fill(0.0)


class BatchSequencesLossNN(CoreNN):
    """Encapsulates one or more NN layers where the output of the top layer is a scalar loss function.

    The input is a batch of sequences, where each sequence can have a different length.

    First dimension is time, second is sequence index.

    It is allowed that this object is only the loss layer or also encapsulates further objects of type
    BatchSequencesComponentNN.
    """

    def __init__(self, num_p, dtype):
        super(BatchSequencesLossNN, self).__init__(num_p, dtype)
        # when this object is a single scalar layer, the following 3 are needed for gradient checks, otherwise they
        # are not necessary as they are part of the enclosed component BatchSequencesComponentNN objects.
        self._seq_lengths = None
        self._x = None
        self._y_true = None

    @abstractmethod
    def forward_backwards(self, x, y_true, seq_lengths):
        """ Performs one forward propagation and one back propagation using the current model.

        The contract regarding values after the sequence length is same as BatchSequencesComponentNN.forward(..) and
        BatchSequencesComponentNN.backwards(..)

        It can't assume anything for values in y_true after end of each sequence. Method is allowed to overwrite them
        to any value, which will be visible to the caller.

        Args:
            x: numpy.array of shape (maxT, N, ..)
                N input sequences, each at most maxT observations
            seq_lengths: numpy.array of type np.integer of shape (N, )
                the lengths of the N input sequences
            y_true: target (true) values, numpy.array
                for classification problems it is usually type np.integer and shape (maxT, N)
                Cannot assume specific values after end of sequence! Can overwrite.
        Returns:
            loss: total loss over all valid observations, scalar
            gradient: gradient of the loss wrt the current parameters, numpy.array of shape (parameter_dimension, )
            delta_error: error vector from this layer, i.e. derivative w.r.to the inputs to this layer,
                None if not-differentiable, numpy.array of shape (maxT, N, ..) (or (N, maxT, ..))
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


def validate_x_and_lengths(x, max_seq_length, max_num_sequences, seq_lengths):
    """Implementation-level code useful for several implementations operating on batched sequences.

    Validates that tensor x has dimensionality a subset of [max_seq_length, max_num_sequences, ...] and that the
    sequence lengths passed fit in the dimensions of x
    """
    assert 2 <= x.ndim
    assert x.shape[0] <= max_seq_length and x.shape[1] <= max_num_sequences
    assert seq_lengths.shape == (x.shape[1],)
    assert np.issubdtype(seq_lengths.dtype, np.integer)
    assert 0 <= seq_lengths.min()
    assert seq_lengths.max() <= x.shape[0]


def glorot_init(shape):
    limit = np.sqrt(6.0 / (np.sum(shape)))
    return np.random.uniform(low=-limit, high=limit, size=shape)

