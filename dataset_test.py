import unittest

import numpy as np

import dataset as dst


class TestDatasetBoundaries(unittest.TestCase):

    def test_with_boundaries(self):
        data = np.array([[0, 2, 3], [1, 4, 1], [2, 6, 7], [3, 1, 3], [4, 5, 7], [5, 1, 1]])
        labels = np.array([2, 1, 1, 0, 2, 0])
        doc_ends = np.array([2, 3, 5])

        dataset = dst.InMemoryDataSetWithBoundaries(data, labels, doc_ends)

        batch_size = 2
        offset1, offset2 = 0, 0

        data1, labels1, boundary = dataset.get_n_next(batch_size)
        offset2 += len(data1)
        self.assertEqual(offset2, 2)  # first doc partial
        self.assertFalse(boundary)
        assert np.alltrue(np.equal(data1, data[offset1:offset2]))
        assert np.alltrue(np.equal(labels1, labels1[offset1:offset2]))

        offset1 = offset2
        data1, labels1, boundary = dataset.get_n_next(batch_size)
        offset2 += len(data1)
        self.assertEqual(offset2, 3)  # first doc fully read
        self.assertTrue(boundary)
        assert np.alltrue(np.equal(data1, data[offset1:offset2]))
        assert np.alltrue(np.equal(labels1, labels1[offset1:offset2]))

        offset1 = offset2
        data1, labels1, boundary = dataset.get_n_next(batch_size)
        offset2 += len(data1)
        self.assertEqual(offset2, 4)  # second doc fully read
        self.assertTrue(boundary)
        assert np.alltrue(np.equal(data1, data[offset1:offset2]))
        assert np.alltrue(np.equal(labels1, labels[offset1:offset2]))

        offset1 = offset2
        data1, labels1, boundary = dataset.get_n_next(batch_size)
        offset2 += len(data1)
        self.assertEqual(offset2, 6)  # third doc fully read
        self.assertTrue(boundary)
        assert np.alltrue(np.equal(data1, data[offset1:offset2]))
        assert np.alltrue(np.equal(labels1, labels[offset1:offset2]))

        offset1, offset2 = 0, 0
        data1, labels1, boundary = dataset.get_n_next(batch_size)
        offset2 += len(data1)
        self.assertEqual(offset2, 2)  # first doc partial
        self.assertFalse(boundary)
        assert np.alltrue(np.equal(data1, data[offset1:offset2]))
        assert np.alltrue(np.equal(labels1, labels[offset1:offset2]))

        offset1 = offset2
        data1, labels1, boundary = dataset.get_n_next(batch_size)
        offset2 += len(data1)
        self.assertEqual(offset2, 3)  # first doc fully read
        self.assertTrue(boundary)
        assert np.alltrue(np.equal(data1, data[offset1:offset2]))
        assert np.alltrue(np.equal(labels1, labels1[offset1:offset2]))


if __name__ == "__main__":
    unittest.main()
