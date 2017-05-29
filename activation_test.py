import unittest
import activation as ac

import numpy as np


class TestDataManip(unittest.TestCase):

    def test_sigmoid(self):
        rtol, atol = 1e-7, 1e-7

        x = np.array([1, 2, 3], dtype=np.float32)
        expected_f = [0.73105858, 0.88079708, 0.9525741]
        expected_g = [0.19661193, 0.10499359, 0.0451766]

        f = ac.sigmoid(x)
        g = ac.sigmoid_grad(f)
        assert f.shape == (3,) and g.shape == (3,)
        assert np.allclose(f, np.array(expected_f), rtol=rtol, atol=atol)
        assert np.allclose(g, np.array(expected_g), rtol=rtol, atol=atol)

        x = np.array([[1, 2, 3]], dtype=np.float32)
        f = ac.sigmoid(x)
        f1 = np.empty_like(f)
        ac.sigmoid(x, out=f1)
        assert np.alltrue(np.equal(f, f1))
        g = ac.sigmoid_grad(f)
        g1 = np.empty_like(g)
        ac.sigmoid_grad(f, out=g1)
        assert np.alltrue(np.equal(g, g1))
        assert f.shape == (1, 3) and g.shape == (1, 3)
        assert np.allclose(f, np.array(expected_f), rtol=rtol, atol=atol)
        assert np.allclose(g, np.array(expected_g), rtol=rtol, atol=atol)

        x = np.array([[1, 2, 3], [-1, -2, -3]], dtype=np.float32)
        expected_f1 = np.ones(len(expected_f)) - expected_f

        f = ac.sigmoid(x)
        f1 = np.empty_like(f)
        ac.sigmoid(x, out=f1)
        assert np.alltrue(np.equal(f, f1))
        g = ac.sigmoid_grad(f)
        g1 = np.empty_like(g)
        ac.sigmoid_grad(f, out=g1)
        assert np.alltrue(np.equal(g, g1))
        assert f.shape == (2, 3) and g.shape == (2, 3)
        assert np.allclose(f, np.array([expected_f, expected_f1]), rtol=rtol, atol=atol)
        assert np.allclose(g, np.array([expected_g, expected_g]), rtol=rtol, atol=atol)

    def test_relu(self):

        x = np.array([1.5, -1, 3], dtype=np.float32)
        expected_f = [1.5, 0, 3]
        expected_g = [1, 0, 1]

        f = ac.relu(x)
        f1 = np.empty_like(f)
        ac.relu(x, out=f1)
        assert np.alltrue(np.equal(f, f1))
        g = ac.relu_grad(f)
        g1 = np.empty_like(g)
        ac.relu_grad(f, out=g1)
        assert np.alltrue(np.equal(g, g1))
        assert f.shape == (3, ) and g.shape == (3, )
        assert np.alltrue(np.equal(f, np.array(expected_f)))
        assert np.alltrue(np.equal(g, np.array(expected_g)))

        x = np.array([x, [-7, 0, 10]], dtype=np.float32)
        expected_f1 = np.array([expected_f, [0, 0, 10]], dtype=np.float32)
        expected_g1 = np.array([expected_g, [0, 0, 1]], dtype=np.float32)

        f = ac.relu(x)
        f1 = np.empty_like(f)
        ac.relu(x, out=f1)
        assert np.alltrue(np.equal(f, f1))
        g = ac.relu_grad(f)
        g1 = np.empty_like(g)
        ac.relu_grad(f, out=g1)
        assert np.alltrue(np.equal(g, g1))
        assert f.shape == (2, 3) and g.shape == (2, 3)
        assert np.alltrue(np.equal(f, np.array(expected_f1)))
        assert np.alltrue(np.equal(g, np.array(expected_g1)))


if __name__ == "__main__":
    unittest.main()
