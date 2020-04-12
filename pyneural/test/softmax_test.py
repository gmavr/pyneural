import unittest

import numpy as np

from pyneural.softmax import softmax_1d, softmax_1d_opt, softmax_2d, softmax_2d_opt, softmax_3d, softmax_3d_opt


class SoftMaxTest(unittest.TestCase):
    def test_softmax(self):
        # softmax(np.array([1, 2])
        expected0 = [0.26894142, 0.73105858]

        # softmax(np.array([1, 2, 3])
        expected1 = [0.09003057, 0.2447285, 0.665241]

        # softmax(np.array([1, 3, 5])
        expected2 = [0.01587624, 0.1173104, 0.8668133]

        rtol = 1e-6
        atol = 1e-10

        dtype = np.float32

        x = np.array([1001.0, 1002.0], dtype=dtype)
        y = softmax_1d(x)
        y1 = softmax_1d_opt(x)
        y2 = np.empty_like(y)
        softmax_1d_opt(x, out=y2)
        self.assertTrue(np.allclose(y, np.array(expected0), rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(y, y1, rtol=rtol, atol=atol))
        self.assertTrue(np.all(np.equal(y1, y2)))
        self.assertEqual(y.shape, x.shape)

        y = softmax_2d(np.array([[1001.0, 1002.0], [3.0, 4.0]], dtype=dtype))
        self.assertTrue(np.allclose(y, np.array([expected0, expected0], dtype=dtype), rtol=rtol, atol=atol))
        self.assertEqual(y.shape, (2, 2))

        x = np.array([[-1001.0, -1002.0]], dtype=dtype)
        y = softmax_2d(x)
        self.assertTrue(np.allclose(y, np.array([expected0[1], expected0[0]], dtype=dtype), rtol=rtol, atol=atol))
        self.assertEqual(y.shape, (1, 2))

        x = np.array([1, 2, 3], dtype=dtype)
        y = softmax_1d(x)
        self.assertTrue(np.allclose(y, np.array(expected1, dtype=dtype), rtol=rtol, atol=atol))
        self.assertEqual(y.shape, (3,))

        x = np.array([[1, 2, 3]], dtype=dtype)
        y = softmax_2d(x)
        self.assertTrue(np.allclose(y, np.array(expected1, dtype=dtype), rtol=rtol, atol=atol))
        self.assertEqual(y.shape, (1, 3))

        x = np.array([[1, 2, 3], [1, 3, 5]], dtype=dtype)
        y = softmax_2d(x)
        y1 = softmax_2d_opt(x)
        y2 = np.empty_like(y)
        softmax_2d_opt(x, out=y2)
        self.assertTrue(np.allclose(y, np.array([expected1, expected2], dtype=dtype), rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(y, y1, rtol=rtol, atol=atol))
        self.assertTrue(np.all(np.equal(y1, y2)))
        self.assertEqual(y.shape, (2, 3))

        x = np.array([[1, 2, 3], [1, 3, 5], [11, 12, 13], [21, 23, 25]], dtype=dtype)
        y = softmax_2d(x)
        y1 = softmax_2d_opt(x)
        y2 = np.empty_like(y)
        softmax_2d_opt(x, out=y2)
        self.assertTrue(np.allclose(y, np.array([expected1, expected2, expected1, expected2],
                                                dtype=dtype), rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(y, y1, rtol=rtol, atol=atol))
        self.assertTrue(np.all(np.equal(y1, y2)))
        self.assertEqual(y.shape, (4, 3))

        y = softmax_1d(np.array([1, 1, 1, 1]))
        self.assertTrue(np.allclose(y, np.array([0.25, 0.25, 0.25, 0.25], dtype=dtype), rtol=1e-20, atol=1e-20))
        self.assertEqual(y.shape, (4,))

        y = softmax_2d(np.array([[100]], dtype=dtype))
        self.assertTrue(np.allclose(y, np.array([1.0], dtype=dtype), rtol=1e-20, atol=1e-20))
        self.assertEqual(y.shape, (1, 1))

        x = np.empty((3, 2, 3), dtype=dtype)
        x[0] = [[1, 2, 3], [1, 3, 5]]
        x[1] = [[1001, 1003, 1005], [1, 1, 1]]
        x[2] = [[5, 5, 5], [11, 12, 13]]

        expected = np.empty((3, 2, 3))
        expected[0] = np.array([expected1, expected2])
        expected[1] = np.array([expected2, [0.3333333, 0.3333333, 0.3333333]])
        expected[2] = np.array([[0.3333333, 0.3333333, 0.3333333], expected1])

        y = softmax_3d(x)
        y1 = softmax_3d_opt(x)
        y2 = np.empty_like(y)
        softmax_3d_opt(x, out=y2)
        self.assertTrue(np.allclose(y, expected, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(y, y1, rtol=rtol, atol=atol))
        self.assertTrue(np.all(np.equal(y1, y2)))
        self.assertEqual(y.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
