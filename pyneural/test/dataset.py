import numpy as np

import pyneural.neural_base as nb

"""
Utility classes for exposing data sets as streams to be used together with neural_base.LossNN objects to fit models.
"""


class InMemoryDataSet:
    def __init__(self, data, labels):
        """
        Args:
            data: dense vectors (data.ndim == 2) or indices in vocabulary (data.ndim == 1)
                In either case, the first dimension indexes the items.
            labels: true values, vector of shape (data.shape[0], )
        """
        assert data.ndim == 2 or data.ndim == 1
        assert labels.ndim == 1
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels
        self.index = 0
        self.data_size = len(self.data)
        self.num_epochs = 0

    def get_n_next(self, num_samples):
        """
        Args:
            num_samples: number of sequential items to retrieve. Must align exactly with size of data.
        Returns:
            data, labels
        """
        assert self.index + num_samples <= self.data_size
        data1 = self.data[self.index:(self.index + num_samples)]
        labels1 = self.labels[self.index:(self.index + num_samples)]
        self.index += num_samples
        if self.index == self.data_size:
            self.index = 0
            self.num_epochs += 1
        return data1, labels1


class InMemoryDataSetWithBoundaries:
    def __init__(self, data, labels, doc_ends):
        """ Initialize data set with M documents consisting of N in total sequential items.
        Args:
            data: sequence of D-dimensional items of all documents appended serially. numpy array of shape (N, D).
            labels: labels of each item. numpy array of shape (N, ).
            doc_ends: indices in data of the last items of each document. numpy array of shape (M, ).
                so data[doc_ends[i]] is the last item of the i-th document
        """
        assert data.ndim == 2
        assert labels.ndim == 1
        assert len(data) == len(labels)
        assert doc_ends.ndim == 1
        assert doc_ends[len(doc_ends)-1] + 1 == len(data)
        self.data = data
        self.labels = labels
        self.doc_ends = doc_ends
        for i in range(len(doc_ends)-1):
            assert doc_ends[i] < doc_ends[i+1]

        self.index = 0  # 0-based index of one past last read
        self.doc_ends_index = 0  # index in doc_ends (points to the end of current document)
        self.data_size = len(self.data)
        self.num_epochs = 0

    def get_n_next(self, num_samples):
        """ Returns up to num_samples from the current position.
        Returns exactly num_samples if the end of current document or end of corpus is not reached, otherwise returns
        enough to reach the end of current document or end of corpus.
        Args:
            num_samples: number of samples requested
        Returns:
            data: slice of dataset
            labels: slice of labels
            boundary_found: whether of not a boundary was met, including the case where the last sample is the last
            sample in the doc
        """
        boundary_found = False
        if self.data_size <= self.index + num_samples:
            # less read (or equal) due to end of data
            num_samples = self.data_size - self.index
            boundary_found = True
        if self.doc_ends[self.doc_ends_index] <= self.index + num_samples - 1:
            # less read (or equal) due to end of doc
            num_samples = self.doc_ends[self.doc_ends_index] - self.index + 1
            self.doc_ends_index += 1
            boundary_found = True
        data1 = self.data[self.index:(self.index + num_samples)]
        labels1 = self.labels[self.index:(self.index + num_samples)]
        self.index += num_samples
        if self.index == self.data_size:
            assert boundary_found  # must have been set
            # wrap-around
            self.index = 0
            self.doc_ends_index = 0
            self.num_epochs += 1
        return data1, labels1, boundary_found


class LossNNAndDataSet:
    def __init__(self, loss_nn, data_set, batch_size):
        assert isinstance(loss_nn, nb.LossNN)
        assert isinstance(data_set, InMemoryDataSet)
        assert data_set.data_size % batch_size == 0
        assert loss_nn.get_max_seq_length() >= batch_size  # XXX why is that even needed? (changed from == to >=)
        self.data_set = data_set
        self.loss_nn = loss_nn
        self.batch_size = batch_size

    def forward_backward_batch(self):
        data, labels = self.data_set.get_n_next(self.batch_size)
        loss, gradient, _ = self.loss_nn.forward_backwards(data, labels)
        # normalize per batch size
        return loss / self.batch_size, gradient / self.batch_size


class LossNNAndDataSetWithBoundary:
    def __init__(self, loss_nn, data_set, batch_size):
        """
        Args:
            loss_nn: An object exposing method reset_last_hidden(), derived class of LossNN
            data_set: object of type InMemoryDataSetWithBoundaries encapsulating the data set item sequences
            batch_size: the number of sequential items from the data stream to be extracted and used in one invocation
                of loss_rnn
        """
        assert isinstance(data_set, InMemoryDataSetWithBoundaries)
        assert isinstance(loss_nn, nb.LossNN)
        assert batch_size > 0
        assert data_set.data_size % batch_size == 0
        assert loss_nn.get_max_seq_length() == batch_size
        self.data_set = data_set
        self.loss_nn = loss_nn
        self.batch_size = batch_size

    def forward_backward_batch(self):
        """
        Sets the supplied model parameters, then reads self.batch_size sequential samples from the data set and performs
        one forward and back-propagation pass on that data. Data consists of sequences of different lengths. When a
        sequence ends and another one starts (either inside the same batch or at a batch boundary), the hidden state
        is reset at the sequence boundaries. Back-propagation is applied to be consistent with the hidden state reset.
        Returns:
            loss, gradient: both divided by self.batch_size (see inline comments for rational)
        """
        loss = np.float64(0.0)
        gradient = np.zeros((1, ), dtype=self.loss_nn.get_dtype())

        # If we reset the hidden state at time t, it is not hard to see from the backpropagation equations that all
        # future states contribute zero error to time t. One way to implement that is to simply split the current batch
        # into sequential batches split at the positions where the hidden state is reset and then set the final
        # gradient to the sum of the gradients for each newly chopped batch.
        #
        num_to_read = self.batch_size
        while num_to_read > 0:
            data, labels, boundary_found = self.data_set.get_n_next(num_to_read)
            loss1, gradient1, _ = self.loss_nn.forward_backwards(data, labels)
            loss += loss1
            gradient = gradient1 + gradient  # note: gradient += gradient1 fails broadcasting
            if boundary_found:
                self.loss_nn.reset_last_hidden()
            num_to_read -= len(data)

        # normalize per batch size
        return loss / self.batch_size, gradient / self.batch_size


def test_with_boundaries():
    data = np.array([[0, 2, 3], [1, 4, 1], [2, 6, 7], [3, 1, 3], [4, 5, 7], [5, 1, 1]])
    labels = np.array([2, 1, 1, 0, 2, 0])
    doc_ends = np.array([2, 3, 5])

    dataset = InMemoryDataSetWithBoundaries(data, labels, doc_ends)

    batch_size = 2
    offset1, offset2 = 0, 0

    data1, labels1, boundary = dataset.get_n_next(batch_size)
    offset2 += len(data1)
    assert offset2 == 2  # first doc partial
    assert not boundary
    assert np.all(np.equal(data1, data[offset1:offset2]))
    assert np.all(np.equal(labels1, labels1[offset1:offset2]))

    offset1 = offset2
    data1, labels1, boundary = dataset.get_n_next(batch_size)
    offset2 += len(data1)
    assert offset2 == 3  # first doc fully read
    assert boundary
    assert np.all(np.equal(data1, data[offset1:offset2]))
    assert np.all(np.equal(labels1, labels1[offset1:offset2]))

    offset1 = offset2
    data1, labels1, boundary = dataset.get_n_next(batch_size)
    offset2 += len(data1)
    assert offset2 == 4  # second doc fully read
    assert boundary
    assert np.all(np.equal(data1, data[offset1:offset2]))
    assert np.all(np.equal(labels1, labels[offset1:offset2]))

    offset1 = offset2
    data1, labels1, boundary = dataset.get_n_next(batch_size)
    offset2 += len(data1)
    assert offset2 == 6  # third doc fully read
    assert boundary
    assert np.all(np.equal(data1, data[offset1:offset2]))
    assert np.all(np.equal(labels1, labels[offset1:offset2]))

    offset1, offset2 = 0, 0
    data1, labels1, boundary = dataset.get_n_next(batch_size)
    offset2 += len(data1)
    assert offset2 == 2  # first doc partial
    assert not boundary
    assert np.all(np.equal(data1, data[offset1:offset2]))
    assert np.all(np.equal(labels1, labels[offset1:offset2]))

    offset1 = offset2
    data1, labels1, boundary = dataset.get_n_next(batch_size)
    offset2 += len(data1)
    assert offset2 == 3  # first doc fully read
    assert boundary
    assert np.all(np.equal(data1, data[offset1:offset2]))
    assert np.all(np.equal(labels1, labels1[offset1:offset2]))


if __name__ == "__main__":
    # Executing this as a standalone script fails without modifications in package resolution.
    # However, running it from within the pycharm IDE succeeds.
    test_with_boundaries()
