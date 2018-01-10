import numpy as np

import bidir_rnn_layer
import ce_softmax_layer as csl
import gru_layer as gru
import rnn_layer as rl
from ce_softmax_layer import CESoftmaxLayerBatch
from embedding_layer import EmbeddingLayerBatch, EmbeddingLayer
from neural_base import LossNN, BatchSequencesLossNN
from neural_layer import NeuralLayer
from rnn_batch_layer import RnnBatchLayer


""" Collection of networks that include RnnLayer, RnnBatchLayer, GruLayer etc.

For demonstrating assembly of multi-layer networks and for testing components.
"""


class HiddenSoftMax(LossNN):
    """ The most elementary classification network: Single hidden layer and softmax top.
    """
    def __init__(self, dim_d, dim_h, dim_k, dtype=np.float64, activation="tanh", asserts_on=True):
        self.dim_d, self.dim_h, self.dim_k = dim_d, dim_h, dim_k
        self.ce_sm = csl.CESoftmaxLayer(self.dim_k, self.dim_h, dtype)
        self.nl = NeuralLayer(self.dim_d, self.dim_h, dtype, activation, asserts_on=asserts_on)
        num_params = self.nl.get_num_p() + self.ce_sm.get_num_p()
        super(HiddenSoftMax, self).__init__(num_params, dtype)
        self.delta_error = None

    def get_display_dict(self):
        d = self._init_display_dict()
        d["ce_sm"] = self.ce_sm.get_display_dict()
        d["nl"] = self.nl.get_display_dict()
        return d

    def get_dimensions(self):
        return self.dim_d, self.dim_h, self.dim_k

    @staticmethod
    def get_max_seq_length():
        return 100000

    def model_normal_init(self, sd):
        assert self._model is not None
        self.nl.model_normal_init(sd)
        self.ce_sm.model_normal_init(sd)

    def model_glorot_init(self):
        assert self._model is not None
        self.nl.model_glorot_init()
        self.ce_sm.model_glorot_init()

    def forward_backwards(self, data, labels):
        self._x, self._y_true = data, labels

        hs = self.nl.forward(data)
        loss = self.ce_sm.forward(hs, labels)

        delta_err_ce_sm = self.ce_sm.backwards()
        self.delta_error = self.nl.backwards(delta_err_ce_sm)

        return loss, self.get_gradient(), self.delta_error

    def _set_model_references_in_place(self):
        params = self._model
        self.nl.set_model_storage(params[0:self.nl.get_num_p()])
        self.ce_sm.set_model_storage(params[self.nl.get_num_p():self._num_p])

    def _set_gradient_references_in_place(self):
        grad = self._grad
        self.nl.set_gradient_storage(grad[0:self.nl.get_num_p()])
        self.ce_sm.set_gradient_storage(grad[self.nl.get_num_p():self._num_p])

    def get_built_model(self):
        return np.concatenate((self.nl.get_model(), self.ce_sm.get_model()))

    def get_built_gradient(self):
        return np.concatenate((self.nl.get_gradient(), self.ce_sm.get_gradient()))

    @staticmethod
    def get_num_p_static(dim_d, dim_h, dim_k):
        return dim_h * (dim_d + 1) + dim_k * (dim_h + 1)


class RnnSoftMax(LossNN):
    def __init__(self, dimensions, max_seq_length, bptt_steps=None, dtype=np.float64, activation="tanh",
                 cell_type="basic", asserts_on=True, grad_clip_thres=None):
        self.dim_d, self.dim_h, self.dim_k = (dimensions[0], dimensions[1], dimensions[2])
        self.ce_sm = csl.CESoftmaxLayer(self.dim_k, self.dim_h, dtype)
        if bptt_steps is None:
            bptt_steps = max_seq_length
        if cell_type == "basic":
            self.rnn = rl.RnnLayer(self.dim_d, self.dim_h, max_seq_length, dtype, activation=activation,
                                   bptt_steps=bptt_steps, grad_clip_thres=grad_clip_thres, asserts_on=asserts_on)
        elif cell_type == "gru":
            self.rnn = gru.GruLayer(self.dim_d, self.dim_h, max_seq_length, dtype=dtype, asserts_on=asserts_on)
        else:
            raise ValueError("Invalid cell type")
        num_params = self.rnn.get_num_p() + self.ce_sm.get_num_p()
        super(RnnSoftMax, self).__init__(num_params, dtype)
        self.delta_error = None

    def get_display_dict(self):
        d = self._init_display_dict()
        d["ce_sm"] = self.ce_sm.get_display_dict()
        d["rnn"] = self.rnn.get_display_dict()
        return d

    def get_dimensions(self):
        return self.dim_d, self.dim_h, self.dim_k

    def set_init_h(self, h_init):
        self.rnn.set_init_h(h_init)

    def get_last_seq_length(self):
        return self.rnn.get_last_seq_length()

    def get_max_seq_length(self):
        return self.rnn.get_max_seq_length()

    def reset_last_hidden(self):
        self.rnn.reset_last_hidden()

    def model_normal_init(self, sd):
        assert self._model is not None
        self.rnn.model_normal_init(sd)
        self.ce_sm.model_normal_init(sd)

    def model_glorot_init(self):
        assert self._model is not None
        self.rnn.model_glorot_init()
        self.ce_sm.model_glorot_init()

    def model_identity_glorot_init(self, scale_factor=0.5):
        assert self._model is not None
        if isinstance(self.rnn, gru.GruLayer):
            raise ValueError("This initialization not supported for GruLayer")
        self.rnn.model_identity_glorot_init(scale_factor)
        self.ce_sm.model_glorot_init()

    def forward_backwards(self, data, labels):
        self._x, self._y_true = data, labels

        hs = self.rnn.forward(data)
        loss = self.ce_sm.forward(hs, labels)

        delta_err_ce_sm = self.ce_sm.backwards()
        self.delta_error = self.rnn.backwards(delta_err_ce_sm)

        return loss, self.get_gradient(), self.delta_error

    def forward_backwards_grad_model(self, **kwargs):
        self.set_init_h(kwargs["h_init"])
        return super(RnnSoftMax, self).forward_backwards_grad_model()

    def forward_backwards_grad_input(self, **kwargs):
        # set the (same) init, because after each forward propagation it is overwritten
        self.set_init_h(kwargs["h_init"])
        return super(RnnSoftMax, self).forward_backwards_grad_input()

    def _set_model_references_in_place(self):
        params = self._model
        self.rnn.set_model_storage(params[0:self.rnn.get_num_p()])
        self.ce_sm.set_model_storage(params[self.rnn.get_num_p():self._num_p])

    def _set_gradient_references_in_place(self):
        grad = self._grad
        self.rnn.set_gradient_storage(grad[0:self.rnn.get_num_p()])
        self.ce_sm.set_gradient_storage(grad[self.rnn.get_num_p():self._num_p])

    def get_built_model(self):
        return np.concatenate((self.rnn.get_model(), self.ce_sm.get_model()))

    def get_built_gradient(self):
        return np.concatenate((self.rnn.get_gradient(), self.ce_sm.get_gradient()))


class RnnEmbeddingsSoftMax(LossNN):
    def __init__(self, rnn_softmax, embedding_layer):
        assert rnn_softmax.get_dtype() == embedding_layer.get_dtype()
        assert isinstance(rnn_softmax, RnnSoftMax)
        assert isinstance(embedding_layer, EmbeddingLayer)
        self.rnn_softmax = rnn_softmax
        self.embedding_layer = embedding_layer
        num_params = rnn_softmax.get_num_p() + embedding_layer.get_num_p()
        super(RnnEmbeddingsSoftMax, self).__init__(num_params, embedding_layer.get_dtype())

    def get_display_dict(self):
        d = self._init_display_dict()
        d["rnn_softmax"] = self.rnn_softmax.get_display_dict()
        d["embedding_layer"] = self.embedding_layer.get_display_dict()
        return d

    def get_dimensions(self):
        return self.rnn_softmax.get_dimensions()

    def get_last_seq_length(self):
        return self.rnn_softmax.get_last_seq_length()

    def get_max_seq_length(self):
        return self.rnn_softmax.get_max_seq_length()

    def forward_backwards(self, data, labels):
        self._x, self._y_true = data, labels

        y = self.embedding_layer.forward(data)
        loss, _, error_vect = self.rnn_softmax.forward_backwards(y, labels)
        self.embedding_layer.backwards(error_vect)
        return loss, self.get_gradient(), None

    def forward_backwards_grad_model(self, **kwargs):
        self.rnn_softmax.set_init_h(kwargs["h_init"])
        return super(RnnEmbeddingsSoftMax, self).forward_backwards_grad_model()

    def forward_backwards_grad_input(self, **kwargs):
        raise ValueError("Derivative w.r. to discrete inputs is not defined")

    def _set_model_references_in_place(self):
        params = self._model
        rnn_num_params = self.rnn_softmax.get_num_p()
        self.rnn_softmax.set_model_storage(params[0:rnn_num_params])
        self.embedding_layer.set_model_storage(params[rnn_num_params:])

    def _set_gradient_references_in_place(self):
        grad = self._grad
        rnn_num_params = self.rnn_softmax.get_num_p()
        self.rnn_softmax.set_gradient_storage(grad[0:rnn_num_params])
        self.embedding_layer.set_gradient_storage(grad[rnn_num_params:])

    def get_built_model(self):
        return np.concatenate((self.rnn_softmax.get_model(), self.embedding_layer.get_model()))

    def get_built_gradient(self):
        return np.concatenate((self.rnn_softmax.get_gradient(), self.embedding_layer.get_gradient()))


class RnnEmbeddingsSoftMaxBatch(BatchSequencesLossNN):
    """ Network of embedding, rnn and softmax operating on batches of sequences.

    Combines 3 out of 4 components currently supporting batches of sequences.
    """

    def __init__(self, softmax_layer, rnn_layer, embedding_layer):
        assert isinstance(softmax_layer, CESoftmaxLayerBatch)
        assert isinstance(rnn_layer, RnnBatchLayer)
        assert isinstance(embedding_layer, EmbeddingLayerBatch)
        num_params = softmax_layer.get_num_p() + rnn_layer.get_num_p() + embedding_layer.get_num_p()
        super(RnnEmbeddingsSoftMaxBatch, self).__init__(
            num_params, rnn_layer.get_max_seq_length(), rnn_layer.get_max_batch_size(),
            rnn_layer.get_dtype())
        self.softmax_layer = softmax_layer
        self.rnn_layer = rnn_layer
        self.embedding_layer = embedding_layer
        assert self._dtype == softmax_layer.get_dtype() and self._dtype == rnn_layer.get_dtype() \
            and self._dtype == embedding_layer.get_dtype()
        assert self._max_seq_length == softmax_layer.get_max_seq_length() \
            and self._max_seq_length == rnn_layer.get_max_seq_length() \
            and self._max_seq_length == embedding_layer.get_max_seq_length()
        assert self._max_num_sequences == softmax_layer.get_max_batch_size() \
            and self._max_num_sequences == rnn_layer.get_max_batch_size() \
            and self._max_num_sequences == embedding_layer.get_max_batch_size()

    def get_display_dict(self):
        d = self._init_display_dict()
        d["softmax_layer"] = self.softmax_layer.get_display_dict()
        d["rnn_layer"] = self.rnn_layer.get_display_dict()
        d["embedding_layer"] = self.embedding_layer.get_display_dict()
        return d

    def forward_backwards(self, x, y_true, seq_lengths):
        self._x, self._y_true, self._seq_lengths = x, y_true, seq_lengths

        y = self.embedding_layer.forward(x, seq_lengths)
        y = self.rnn_layer.forward(y, seq_lengths)
        loss, _, error_vect = self.softmax_layer.forward_backwards(y, y_true, seq_lengths)
        error_vect = self.rnn_layer.backwards(error_vect)
        self.embedding_layer.backwards(error_vect)
        return loss, self.get_gradient(), None

    def forward_backwards_grad_model(self, **kwargs):
        self.rnn_layer.set_init_h(kwargs["h_init"])
        loss, grad, _ = self.forward_backwards(self._x, self._y_true, self._seq_lengths)
        return loss, grad

    def forward_backwards_grad_input(self, **kwargs):
        # set the (same) init, because after each forward propagation it is overwritten
        self.rnn_layer.set_init_h(kwargs["h_init"])
        loss, _, error_vect = self.forward_backwards(self._x, self._y_true, self._seq_lengths)
        return loss, error_vect

    def _set_model_references_in_place(self):
        params = self._model
        p1 = 0
        p2 = self.embedding_layer.get_num_p()
        self.embedding_layer.set_model_storage(params[p1:p2])
        p1 = p2
        p2 += self.rnn_layer.get_num_p()
        self.rnn_layer.set_model_storage(params[p1:p2])
        p1 = p2
        p2 += self.softmax_layer.get_num_p()
        self.softmax_layer.set_model_storage(params[p1:p2])

    def _set_gradient_references_in_place(self):
        grad = self._grad
        p1 = 0
        p2 = self.embedding_layer.get_num_p()
        self.embedding_layer.set_gradient_storage(grad[p1:p2])
        p1 = p2
        p2 += self.rnn_layer.get_num_p()
        self.rnn_layer.set_gradient_storage(grad[p1:p2])
        p1 = p2
        p2 += self.softmax_layer.get_num_p()
        self.softmax_layer.set_gradient_storage(grad[p1:p2])

    def get_built_model(self):
        return np.concatenate((self.embedding_layer.get_model(), self.rnn_layer.get_model(),
                               self.softmax_layer.get_model()))

    def get_built_gradient(self):
        return np.concatenate((self.embedding_layer.get_gradient(), self.rnn_layer.get_gradient(),
                               self.softmax_layer.get_gradient()))


class RnnClassSoftMax(LossNN):
    """ RNN and top class-group softmax layer
      
    Network groups final classes in groups of classes and invokes softmax twice: once to determine the group class and
    given that to determine the final class.
    Initial impl where all classes have same number of words. Good impl should be classes each with same probability.
    FIXME: gradient check broke after transition to latest framework for base classes.
    """
    def __init__(self, dimensions, word_class_mapper, batch_size, dtype=np.float64):
        self.dim_d, self.dim_h, self.dim_k = (dimensions[0], dimensions[1], dimensions[2])
        num_classes = word_class_mapper.get_num_word_classes()
        assert self.dim_k % num_classes == 0
        self.num_classes = num_classes
        self.num_words_per_class = self.dim_k / num_classes
        self.word_class_mapper = word_class_mapper
        self.num_samples = batch_size

        self.rnn = rl.RnnLayer(self.dim_d, self.dim_h, batch_size, dtype)

        self.class_ce_sm = csl.CESoftmaxLayer(self.num_classes, self.dim_h, dtype)
        self.word_ce_sm = [csl.CESoftmaxLayer(self.num_words_per_class, self.dim_h, dtype)
                           for i in xrange(self.num_classes)]

        num_params = self.rnn.get_num_p() + self.class_ce_sm.get_num_p()\
            + self.num_classes * self.word_ce_sm[0].get_num_p()
        self.word_classes_num_param = self.num_classes * self.word_ce_sm[0].get_num_p()

        super(RnnClassSoftMax, self).__init__(num_params, dtype)
        self.delta_error = None

        self.word_classes_grad = np.zeros((self.num_classes, self.word_ce_sm[0].get_num_p()), dtype=self._dtype)
        self.word_classes = np.zeros((self.num_samples, ), dtype=np.int)  # XXX labels.dtype

    def get_display_dict(self):
        d = self._init_display_dict()
        inverted_dict = dict((v, k) for k, v in self.__dict__.items())
        d[inverted_dict[self.rnn]] = self.rnn.get_display_dict()
        d[inverted_dict[self.class_ce_sm]] = self.class_ce_sm.get_display_dict()
        d["num_classes"] = self.num_classes
        return d

    def get_dimensions(self):
        return self.dim_d, self.dim_h, self.dim_k

    def set_init_h(self, h_init):
        self.rnn.set_init_h(h_init)

    def get_last_seq_length(self):
        return self.rnn.get_last_seq_length()

    def get_max_seq_length(self):
        return self.rnn.get_max_seq_length()

    def _set_model_references_in_place(self):
        params = self._model
        ofs1 = 0
        ofs2 = self.rnn.get_num_p()
        self.rnn.set_model_storage(params[ofs1:ofs2])
        ofs1 = ofs2
        ofs2 += self.class_ce_sm.get_num_p()
        self.class_ce_sm.set_model_storage(params[ofs1:ofs2])
        for i in xrange(self.num_classes):
            ofs1 = ofs2
            ofs2 += self.word_ce_sm[i].get_num_p()
            self.word_ce_sm[i].set_model_storage(params[ofs1:ofs2])

    def _set_gradient_references_in_place(self):
        grad = self._grad
        ofs1 = 0
        ofs2 = self.rnn.get_num_p()
        self.rnn.set_gradient_storage(grad[ofs1:ofs2])
        ofs1 = ofs2
        ofs2 += self.class_ce_sm.get_num_p()
        self.class_ce_sm.set_gradient_storage(grad[ofs1:ofs2])
        self.word_classes_grad = np.reshape(grad[ofs2:self.get_num_p()],
                                            (self.num_classes, self.word_ce_sm[0].get_num_p()))
        for i in xrange(self.num_classes):
            ofs1 = ofs2
            ofs2 += self.word_ce_sm[i].get_num_p()
            self.word_ce_sm[i].set_gradient_storage(grad[ofs1:ofs2])
            assert np.shares_memory(self.word_classes_grad[i], grad[ofs1:ofs2])

        # this is necessary because we overwrite only previous batch's set class gradients, but the first time
        # we have no previous set classes!
        self.word_classes_grad.fill(0.0)

    def forward_backwards(self, data, labels):
        self._x, self._y_true = data, labels

        hs = self.rnn.forward(data)

        word_classes = self.word_classes

        # num_word_nn_params = self.word_ce_sm[0].get_number_parameters()
        for i in xrange(self.num_samples):
            # overwrite (sparse) gradient from previous
            # self.word_classes_grad[word_classes[i]] = np.zeros(num_word_nn_params, )
            self.word_classes_grad[word_classes[i]].fill(0.0)
            word_classes[i] = self.word_class_mapper.get_word_class_for(labels[i])

        loss = self.class_ce_sm.forward(hs, word_classes)
        delta_err = self.class_ce_sm.backwards()

        # This can be optimized. Instead of allocating new, we can remember previous and zero them.
        # Experimenting with similar concept in EmbeddingLayer there is performance improvement only when very sparse.

        for i in xrange(self.num_samples):
            label_in_class = self.word_class_mapper.get_label_in_class_for(labels[i])
            word_class = word_classes[i]

            loss_word, grad_word, delta_err_word\
                = self.word_ce_sm[word_class].forward_backwards_single(hs[i], label_in_class)

            # p(word) = p(class) * P(word|class). The CE loss converts to sum of logs = loss_class + loss_word
            loss += loss_word
            delta_err[i, :] += delta_err_word
            self.word_classes_grad[word_class] += grad_word

        self.delta_error = self.rnn.backwards(delta_err)

        return loss, self.get_gradient(), self.delta_error

    def forward_backwards_grad_model(self, **kwargs):
        self.set_init_h(kwargs["h_init"])
        return super(RnnClassSoftMax, self).forward_backwards_grad_model()

    def forward_backwards_grad_input(self, **kwargs):
        self.set_init_h(kwargs["h_init"])
        return super(RnnClassSoftMax, self).forward_backwards_grad_input()

    def get_built_model(self):
        params = np.empty(self.get_num_p(), dtype=self._dtype)

        ofs1 = 0
        ofs2 = self.rnn.get_num_p()
        params[ofs1:ofs2] = self.rnn.get_model()
        ofs1 = ofs2
        ofs2 += self.class_ce_sm.get_num_p()
        params[ofs1:ofs2] = self.class_ce_sm.get_model()
        for i in xrange(self.num_classes):
            ofs1 = ofs2
            ofs2 += self.word_ce_sm[i].get_num_p()
            params[ofs1:ofs2] = self.word_ce_sm[i].get_model()

        return params

    def get_built_gradient(self):
        grad = np.empty(self.get_num_p(), dtype=self._dtype)

        ofs1 = 0
        ofs2 = self.rnn.get_num_p()
        grad[ofs1:ofs2] = self.rnn.get_gradient()
        ofs1 = ofs2
        ofs2 += self.class_ce_sm.get_num_p()
        grad[ofs1:ofs2] = self.class_ce_sm.get_gradient()
        for i in xrange(self.num_classes):
            ofs1 = ofs2
            ofs2 += self.word_ce_sm[i].get_num_p()
            grad[ofs1:ofs2] = self.word_ce_sm[i].get_gradient()

        # FOLLOWING WORKED CORRECTLY IN A PREVIOUS VERSION
        # ofs1 = ofs2
        # ofs2 += self.word_classes_num_param
        # grad[ofs1:ofs2] = np.reshape(self.word_classes_grad, (self.word_classes_num_param, ))

        return grad


class WordClassMapper:
    def __init__(self, num_classes, dim_k):
        assert dim_k % num_classes == 0
        self.num_classes = num_classes
        self.dim_k = dim_k

    def get_num_word_classes(self):
        return self.num_classes

    def get_word_class_for(self, k):
        return k % self.num_classes

    def get_label_in_class_for(self, k):
        return int(k / self.num_classes)


class BidirRnnSoftMax(LossNN):
    def __init__(self, dimensions, max_seq_length, bptt_steps=None, dtype=np.float32, activation="tanh",
                 cell_type="basic"):
        self.dim_d, self.dim_h, self.dim_k = dimensions
        self.bi_rnn = bidir_rnn_layer.BidirRnnLayer(self.dim_d, self.dim_h, max_seq_length, bptt_steps, dtype,
                                                    activation, cell_type)
        self.ce_sm = csl.CESoftmaxLayer(self.dim_k, 2 * self.dim_h, dtype)
        super(BidirRnnSoftMax, self).__init__(self.bi_rnn.get_num_p() + self.ce_sm.get_num_p(), dtype)
        self.delta_error = None

    def get_display_dict(self):
        d = self._init_display_dict()
        d['bi_rnn'] = self.bi_rnn.get_display_dict()
        d['ce_sm'] = self.ce_sm.get_display_dict()
        return d

    def set_init_h(self, h_init_f):
        self.bi_rnn.set_init_h(h_init_f)

    def get_max_seq_length(self):
        return self.bi_rnn.get_max_seq_length()

    def forward_backwards(self, data, labels):
        self._x, self._y_true = data, labels

        hs = self.bi_rnn.forward(data)
        loss = self.ce_sm.forward(hs, labels)

        delta_err_ce_sm = self.ce_sm.backwards()
        self.delta_error = self.bi_rnn.backwards(delta_err_ce_sm)

        return loss, self.get_gradient(), self.delta_error

    def forward_backwards_grad_model(self, **kwargs):
        self.bi_rnn.set_init_h(kwargs["h_init"])
        return super(BidirRnnSoftMax, self).forward_backwards_grad_model()

    def forward_backwards_grad_input(self, **kwargs):
        # set the (same) init, because after each forward propagation it is overwritten
        self.bi_rnn.set_init_h(kwargs["h_init"])
        return super(BidirRnnSoftMax, self).forward_backwards_grad_input()

    def _set_model_references_in_place(self):
        params = self._model
        self.bi_rnn.set_model_storage(params[0:self.bi_rnn.get_num_p()])
        self.ce_sm.set_model_storage(params[self.bi_rnn.get_num_p():self._num_p])

    def _set_gradient_references_in_place(self):
        grad = self._grad
        self.bi_rnn.set_gradient_storage(grad[0:self.bi_rnn.get_num_p()])
        self.ce_sm.set_gradient_storage(grad[self.bi_rnn.get_num_p():self._num_p])

    def get_built_model(self):
        return np.concatenate((self.bi_rnn.get_model(), self.ce_sm.get_model()))

    def get_built_gradient(self):
        return np.concatenate((self.bi_rnn.get_gradient(), self.ce_sm.get_gradient()))

    @staticmethod
    def get_num_p_static(dim_d, dim_h, dim_k):
        return 2 * dim_h * (dim_d + dim_h + 1) + 2 * dim_h * dim_k + dim_k


class EmbeddingBidirRnnSoftMax(LossNN):
    def __init__(self, dimensions, max_seq_length, bptt_steps=None, dtype=np.float64, activation="tanh"):
        self.word_vocab_size, self.dim_d, self.dim_h, self.dim_k = dimensions
        self.word_em = EmbeddingLayer(self.word_vocab_size, self.dim_d, dtype)
        self.bi_rnn = bidir_rnn_layer.BidirRnnLayer(self.dim_d, self.dim_h, max_seq_length, bptt_steps, dtype,
                                                    activation)
        self.ce_sm = csl.CESoftmaxLayer(self.dim_k, 2 * self.dim_h, dtype)
        num_params = self.word_em.get_num_p() + self.bi_rnn.get_num_p() + self.ce_sm.get_num_p()
        super(EmbeddingBidirRnnSoftMax, self).__init__(num_params, dtype)
        self.delta_error = None

    def get_display_dict(self):
        d = self._init_display_dict()
        d['word_em'] = self.word_em.get_display_dict()
        d['bi_rnn'] = self.bi_rnn.get_display_dict()
        d['ce_sm'] = self.ce_sm.get_display_dict()
        return d

    def set_init_h(self, h_init_f):
        self.bi_rnn.set_init_h(h_init_f)

    def get_max_seq_length(self):
        return self.bi_rnn.get_max_seq_length()

    def forward_backwards(self, data, labels):
        self._x, self._y_true = data, labels

        words = self.word_em.forward(data)
        hs = self.bi_rnn.forward(words)
        loss = self.ce_sm.forward(hs, labels)

        delta_err_ce_sm = self.ce_sm.backwards()
        delta_err_rnn = self.bi_rnn.backwards(delta_err_ce_sm)
        self.word_em.backwards(delta_err_rnn)

        return loss, self.get_gradient(), None

    def forward_backwards_grad_model(self, **kwargs):
        self.bi_rnn.set_init_h(kwargs["h_init"])
        return super(EmbeddingBidirRnnSoftMax, self).forward_backwards_grad_model()

    def forward_backwards_grad_input(self, **kwargs):
        # set the (same) init, because after each forward propagation it is overwritten
        self.bi_rnn.set_init_h(kwargs["h_init"])
        return super(EmbeddingBidirRnnSoftMax, self).forward_backwards_grad_input()

    def _set_model_references_in_place(self):
        params = self._model

        of1 = 0
        of2 = self.word_em.get_num_p()
        self.word_em.set_model_storage(params[of1:of2])
        of1 = of2
        of2 += self.bi_rnn.get_num_p()
        self.bi_rnn.set_model_storage(params[of1:of2])
        of1 = of2
        of2 += self.ce_sm.get_num_p()
        self.ce_sm.set_model_storage(params[of1:of2])
        assert of2 == self._num_p

    def _set_gradient_references_in_place(self):
        grad = self._grad

        of1 = 0
        of2 = self.word_em.get_num_p()
        self.word_em.set_gradient_storage(grad[of1:of2])
        of1 = of2
        of2 += self.bi_rnn.get_num_p()
        self.bi_rnn.set_gradient_storage(grad[of1:of2])
        of1 = of2
        of2 += self.ce_sm.get_num_p()
        self.ce_sm.set_gradient_storage(grad[of1:of2])
        assert of2 == self._num_p

    def get_built_model(self):
        return np.concatenate((self.word_em.get_model(), self.bi_rnn.get_model(), self.ce_sm.get_model()))

    def get_built_gradient(self):
        return np.concatenate((self.word_em.get_gradient(), self.bi_rnn.get_gradient(), self.ce_sm.get_gradient()))
