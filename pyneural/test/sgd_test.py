import unittest

import numpy as np

import pyneural.sgd as sgd


class FuncClass(object):
    """
    Function has local minima: 1 at (-1, -1) and 1 at (1, 1), saddle point at (0, 0)
    """
    def __init__(self):
        self.theta = None

    def set_x(self, x):
        self.theta = x

    def func(self):
        x = self.theta[0]
        y = self.theta[1]
        return (2 * x ** 2 - 4 * x * y + y ** 4 + 2,
                np.array([4 * x - 4 * y, - 4 * x + 4 * y ** 3]))


class TestSgdVariants(unittest.TestCase):

    def do_sgd_3points(self, solver1, solver2, solver3, obj):
        assert type(solver1) is type(solver2) and type(solver1) is type(solver3)
        print("Testing SGD solver %s at 3 start points" % solver1.__class__.__name__)

        self.do_sgd(solver1, obj, [-0.5, -0.5], [-1, -1])

        # trapped at the saddle point, which is an (unstable) equilibrium point
        self.do_sgd(solver2, obj, [0.0, 0.0], [0.0, 0.0])

        # but a tiny bit off the saddle point it follows the downhill
        self.do_sgd(solver3, obj, [0.0, 1e-7], [1, 1])

    def do_sgd(self, solver, obj, start, expected):
        # print("Testing SGD solver %s" % solver.__class__.__name__)

        solver.set_lr(0.02)
        solver.num_epochs = 1000
        solver.num_items = 1
        solver.mini_batch_size = 1
        solver.anneal_every_num_epochs = 10000
        solver.report_every_num_epochs = 250

        x0 = np.array(start, dtype=np.float64)
        obj.set_x(x0)
        solver.set_x(x0)
        solver.sgd(obj.func)

        eps = 1e-7
        self.assertTrue(np.allclose(x0, np.array(expected, dtype=np.float64), eps, eps))

    def test_gradient_with_gc(self):

        obj = FuncClass()

        solver1, solver2, solver3 = sgd.SgdSolverAdam(), sgd.SgdSolverAdam(), sgd.SgdSolverAdam()
        self.do_sgd_3points(solver1, solver2, solver3, obj)

        # in this example momentum_factor = 0.95 performs better than 0.99 for both nag on and off
        # in this example convergence from fastest to slowest: Adam (462), NAG (522), vanilla SGD (841), momentum (973)

        solver1, solver2, solver3 = sgd.SgdSolverMomentum(), sgd.SgdSolverMomentum(), sgd.SgdSolverMomentum()
        solver1.set_momentum_factor(0.95)
        solver2.set_momentum_factor(0.95)
        solver3.set_momentum_factor(0.95)
        self.do_sgd_3points(solver1, solver2, solver3, obj)

        solver1, solver2, solver3 = sgd.SgdSolverMomentumNag(), sgd.SgdSolverMomentumNag(), sgd.SgdSolverMomentumNag()
        solver1.set_momentum_factor(0.95)
        solver2.set_momentum_factor(0.95)
        solver3.set_momentum_factor(0.95)
        self.do_sgd_3points(solver1, solver2, solver3, obj)

        solver1, solver2, solver3 = sgd.SgdSolverStandard(), sgd.SgdSolverStandard(), sgd.SgdSolverStandard()
        self.do_sgd_3points(solver1, solver2, solver3, obj)


if __name__ == "__main__":
    unittest.main()
