import cPickle as pickle
import datetime
import glob
import json
import os.path as op
import random
import time
from abc import ABCMeta, abstractmethod

import numpy as np


class SgdSolver(object):
    __metaclass__ = ABCMeta

    # Practical resources for SGD fitting and SGD variations:
    # http://sebastianruder.com/optimizing-gradient-descent/
    # http://cs231n.github.io/neural-networks-3/  "Parameter updates"
    # Bengio 2012 "Practical Recommendations for Gradient-Based Training of Deep Architectures"
    # https://arxiv.org/pdf/1206.5533v2.pdf
    # Sutskever phd thesis 7.2 p. 74 - 77

    def __init__(self):
        # learning rate, the single most important tuning parameter
        self._lr = 0.01

        # Reporting and setting quantities in terms of the epoch-derived quantities is useful for human observation
        # only. The sgd algorithm itself is only concerned with number of iterations.

        # Number of samples in the training data set.
        # Used for conversion from number of (fractional) epochs to number of iterations.
        # It is required that num_items is integer multiple of mini-batch size, so truncate or 0-pad to integer
        # multiples outside of this class.
        self.num_items = None

        # Number of samples in one mini-batch. Used for conversion from (fractional) epochs to iterations
        # and normalizing the loss reported.
        self.mini_batch_size = None

        # How many epochs to run for. Can be fractional.
        self.num_epochs = 5.0

        # Number of epochs for learning rate to be halved (annealed). Can be fractional.
        # One recommendation for "step decay": reduce learning rate by a half every 5 epochs or by 0.1 every 20 epochs.
        self.anneal_every_num_epochs = None

        self.report_every_num_epochs = 0.1
        self.save_every_num_epochs = None

        # Evaluate the model (forward pass) every this number of epochs
        self.evaluate_every_num_epochs = None

        # self-contained directory where all data is to be saved and read, if so enabled
        self.root_dir = None

        self._smooth_loss = None

        # Quantities in terms of iterations instead of epochs.
        self._report_every = None
        self._save_every = None
        self._evaluate_every = None
        self._anneal_every = None

        # all following are 1-based
        # XXX REMOVE: _last_report_iter: is read only once and it initialized with _start_iteration_index 0-based!
        self._next_report_iter, self._last_report_iter, self._num_reported = 0, 0, 0
        self._next_save_iter, self._num_saved = 0, 0
        self._next_eval_iter, self._num_evaluated = 0, 0
        self._next_anneal_iter, self._num_annealed = 0, 0

        # 0-based index of first iteration (> 0 for starting from saved)
        # this is the only 0-based index
        self._start_iteration_index = 0

        # 1-based index of the last iteration to be executed
        self._last_iteration_index = 0

        self._x = None
        self._objective_func = None
        self._evaluate_func = None

        self._time_mark = None
        # total time spent, excluding storing the model and reporting results
        self._total_time_elapsed = 0.0

        self._initialized = False

    def set_lr(self, lr):
        self._lr = lr

    def get_num_iterations(self):
        return self._last_iteration_index - self._start_iteration_index

    def get_class_fq_name(self):
        return self.__module__ + "." + self.__class__.__name__

    def _init_display_dict(self):
        return {'fq_name': self.get_class_fq_name(),
                'lr': self._lr, 'mini_batch_size': self.mini_batch_size,
                'num_items': self.num_items, 'num_epochs': self.num_epochs,
                'anneal_every_num_epochs': self.anneal_every_num_epochs}

    def set_x(self, x):
        """
        Sets reference to x, an numpy array. It will be updated in-place iteratively by SGD.
        It is required that the function to be optimized holds an internal reference to the same x (i.e. it is a member
        function of an object that holds that state)
        Args:
            x: independent variables, 1-D numpy.array
        """
        assert x.ndim == 1
        assert x.dtype == np.float32 or x.dtype == np.float64
        self._x = x

    def init_from_saved(self, min_iteration_index=None):
        """
        Args:
            min_iteration_index: the saved model's minimum iteration number to load.
                If None, the maximum iteration number saved found.
        Returns:
            x0: The initial point to start SGD from, 1-D numpy.array.
                Updated in-place iteratively by SGD.
                Caller must set this as the vector that the function to be minimized is applied on.
        """
        self._load_sgd_parameters()

        self._start_iteration_index, saved_x0, random_state = self._load_saved_state(min_iteration_index)
        if self._start_iteration_index is None or self._start_iteration_index <= 0:
            raise ValueError('No data for iteration index %d or higher was found in %s'
                             % (min_iteration_index, self.root_dir))
        print('Restart: Loaded data for iteration index %d' % self._start_iteration_index)

        # integer multiple of batch size required, truncate or 0-pad to integer multiples outside of this class
        assert self.num_items % self.mini_batch_size == 0

        num_batches_per_epoch = self.num_items / self.mini_batch_size

        if self._start_iteration_index % num_batches_per_epoch != 0:
            # otherwise data needs to be advanced forward, which we can't do from here
            raise ValueError('stored iteration index must be integer multiple of epoch')

        frac1 = self._start_iteration_index / (self.report_every_num_epochs * num_batches_per_epoch)
        frac2 = (self._start_iteration_index + 1) / (self.report_every_num_epochs * num_batches_per_epoch)
        self._num_reported = max((int(frac1), int(frac2)))

        if self.save_every_num_epochs is not None:
            assert self.root_dir is not None
            frac1 = self._start_iteration_index / (self.save_every_num_epochs * num_batches_per_epoch)
            frac2 = (self._start_iteration_index + 1) / (self.save_every_num_epochs * num_batches_per_epoch)
            self._num_saved = max((int(frac1), int(frac2)))

        if self.evaluate_every_num_epochs is not None:
            assert self.evaluate_every_num_epochs > 0.0
            frac1 = self._start_iteration_index / (self.evaluate_every_num_epochs * num_batches_per_epoch)
            frac2 = (self._start_iteration_index + 1) / (self.evaluate_every_num_epochs * num_batches_per_epoch)
            self._num_evaluated = max((int(frac1), int(frac2)))

        if self.anneal_every_num_epochs is not None:
            assert self.anneal_every_num_epochs > 0.0
            frac1 = self._start_iteration_index / (self.anneal_every_num_epochs * num_batches_per_epoch)
            frac2 = (self._start_iteration_index + 1) / (self.anneal_every_num_epochs * num_batches_per_epoch)
            self._num_annealed = max((int(frac1), int(frac2)))
            if self._num_annealed > 0:
                self._lr *= 0.5 ** self._num_annealed
                print("lr annealed %d times, final value %f" % (self._num_annealed, self._lr))

        assert random_state is not None
        random.setstate(random_state)

        self.set_x(saved_x0)  # for assertions to be applied

        return saved_x0

    def sgd(self, objective_function, evaluate_function=None):
        """ Runs Stochastic Gradient Descent.

        There is no notion of data set, batch size, epoch etc directly used the SGD algorithm other than for counters
        and reporting. These are encapsulated in the object exposing the function to be optimized.

        Args:
            objective_function: scalar function to minimize. Takes no direct arguments, but is expected to evaluate the
                same x vector that SGD updates in-place.
            evaluate_function: function on foreign object to compute loss using the current model. Invoked every
                evaluate_every_num_epochs if set to positive.
        """
        if self._initialized:
            raise ValueError("SgdSolver is single use only")
        self._initialized = True

        # verify valid values passed
        assert self.num_items and self.mini_batch_size
        assert self.num_items >= self.mini_batch_size > 0 and self._lr > 0.0 and self.num_epochs > 0.0

        # integer multiple of batch size required, truncate or 0-pad to integer multiples outside of this class
        assert self.num_items % self.mini_batch_size == 0

        num_batches_per_epoch = self.num_items / self.mini_batch_size
        self._last_iteration_index = int(self.num_epochs * num_batches_per_epoch)

        print('Next iteration index: %d' % (self._start_iteration_index + 1))

        self._report_every = self.report_every_num_epochs * num_batches_per_epoch
        self._next_report_iter = int(self._report_every * (self._num_reported + 1))
        self._last_report_iter = self._start_iteration_index
        print('Next report iteration index: %d, period: %.1f' % (self._next_report_iter, self._report_every))

        if self.save_every_num_epochs is not None:
            assert self.root_dir is not None
            self._save_every = self.save_every_num_epochs * num_batches_per_epoch
            self._next_save_iter = int(self._save_every * (self._num_saved + 1))
            print('Next saving iteration index: %d, period: %.1f' % (self._next_save_iter, self._save_every))

        if evaluate_function is not None:
            assert self.evaluate_every_num_epochs > 0.0
            self._evaluate_every = self.evaluate_every_num_epochs * num_batches_per_epoch
            self._next_eval_iter = int(self._evaluate_every * (self._num_evaluated + 1))
            print('Next evaluation iteration index: %d, period: %.1f' % (self._next_eval_iter, self._evaluate_every))

        if self.anneal_every_num_epochs is not None:
            self._anneal_every = self.anneal_every_num_epochs * num_batches_per_epoch
            self._next_anneal_iter = int(self._anneal_every * (self._num_annealed + 1))
            print('Next lr annealing iteration index: %d, period: %.1f'
                  % (self._next_anneal_iter, self._anneal_every))

        self._objective_func = objective_function
        self._evaluate_func = evaluate_function

        print('Starting SGD. Number iterations per epoch %d. Time: %s'
              % (num_batches_per_epoch, '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))

        json_string = json.dumps(self.get_class_dict(), indent=4, sort_keys=True)
        if self.root_dir:
            self._save_sgd_parameters(json_string)
        else:
            print(json_string)

        self._derived_init()

        self._sgd_loop()

    def _sgd_loop(self):
        # smooth_loss is for reporting purposes only, does not affect algorithm.
        # Since we print the loss periodically, it is more meaningful to print a weighted running average
        # (weighted heavily towards recent values) than the last iteration's loss (which is much more noisy).

        self._time_mark = time.time()

        # iteration_index is 1-based, self._start_iteration_index is 0-based
        for iteration_index in xrange(self._start_iteration_index + 1, self._last_iteration_index + 1):
            loss, update_x = self._get_update_and_loss()

            if np.isinf(loss):
                print("iteration %d: Warning: infinite loss, reporting current smoothed loss", iteration_index)
                loss = self._smooth_loss

            if not self._smooth_loss:
                self._smooth_loss = loss
            else:
                self._smooth_loss = .99 * self._smooth_loss + .01 * loss

            if iteration_index == self._next_report_iter:
                self._report_metrics(iteration_index, loss, update_x)

            self._x -= update_x

            if self._save_every is not None and iteration_index == self._next_save_iter:
                self._save_state(iteration_index)

            if self._evaluate_every is not None and iteration_index == self._next_eval_iter:
                self._evaluate(iteration_index)

            if self._anneal_every is not None and iteration_index == self._next_anneal_iter:
                self._lr *= 0.5
                print("iteration %d: lr annealed to %f" % (iteration_index, self._lr))
                self._num_annealed += 1
                self._next_anneal_iter = int(self._anneal_every * (self._num_annealed + 1))

        time_elapsed = time.time() - self._time_mark
        self._total_time_elapsed += time_elapsed

        print("Finished SGD. Time: %s" % '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        num_executed = self.get_num_iterations()
        print("Finished at %.3f epoch. Processing rate: %.4g sec per batch of size %d, %.2f min per epoch."
              % (self.num_epochs, self._total_time_elapsed / num_executed, self.mini_batch_size,
                 self.num_items * self._total_time_elapsed / (60 * num_executed * self.mini_batch_size)))

    def _evaluate(self, iteration_index):
        self._total_time_elapsed += (time.time() - self._time_mark)

        print("iteration %d: Evaluating on held-out data set. Time: %s"
              % (iteration_index, '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))
        loss = self._evaluate_func()
        print("iteration %d: dev loss: %f current smoothed loss: %f" % (iteration_index, loss, self._smooth_loss))

        self._num_evaluated += 1
        self._next_eval_iter = int(self._evaluate_every * (self._num_evaluated + 1))

        self._time_mark = time.time()

    def _report_metrics(self, iteration_index, loss, update_x):

        time_elapsed = time.time() - self._time_mark

        if self._total_time_elapsed == 0:
            # first time reporting
            per_batch_time_elapsed = float(time_elapsed) / (self._next_report_iter - self._last_report_iter)
            print("per batch time: %.4f sec, per item time: %.4e sec, estimated per epoch time: %.3f hr"
                  % (per_batch_time_elapsed, per_batch_time_elapsed / self.mini_batch_size,
                     (per_batch_time_elapsed * self.num_items / self.mini_batch_size) / 3600.0))

        self._total_time_elapsed += time_elapsed

        param_scale = np.linalg.norm(self._x, 2)
        update_scale = np.linalg.norm(update_x, 2)

        update_ratio = float('inf') if param_scale == 0.0 else update_scale / param_scale

        print("iteration %d: epoch %.2f: smoothed loss: %.4g last loss: %.4g update ratio: %.3e"
              % (iteration_index,
                 (float(iteration_index) * self.mini_batch_size / self.num_items),
                 self._smooth_loss, loss, update_ratio))

        self._last_report_iter = self._next_report_iter
        self._num_reported += 1
        self._next_report_iter = int(self._report_every * (self._num_reported + 1))

        self._time_mark = time.time()

    @abstractmethod
    def get_class_dict(self):
        """
        Returns:
            Dictionary of class elements.
        """

    @abstractmethod
    def _load_from_class_dict(self, dictionary):
        pass

    @abstractmethod
    def _derived_init(self):
        """
        Initialize derived-class internal state (just the dimension really) given the x vector.
        """

    @abstractmethod
    def _get_update_and_loss(self):
        """
        Returns:
            loss: f(x) the output of scalar function f evaluated at vector x
            update_x: the update that should be subtracted from x at the next iteration
        """

    def _save_sgd_parameters(self, json_string):
        name_index = 0
        for f in glob.glob(self.root_dir + "/sgd_parameters_*.json"):
            file_name_no_extension = op.splitext(op.basename(f))[0]
            name_index1 = int(file_name_no_extension.split("_")[2])
            if name_index1 > name_index:
                name_index = name_index1

        name_index += 1

        path_write = '%s/sgd_parameters_%d.json' % (self.root_dir, name_index)
        with open(path_write, 'w') as writer:
            writer.write(json_string)

    def _load_sgd_parameters(self):
        """
        Loads the saved parameters overwriting existing except for a few that are properties of the data set
        (which is assumed to be the same) and required to be a match.
        """
        name_index = None
        for f in glob.glob(self.root_dir + "/sgd_parameters_*.json"):
            file_name_no_extension = op.splitext(op.basename(f))[0]
            name_index1 = int(file_name_no_extension.split("_")[2])
            if name_index is None or name_index1 > name_index:
                name_index = name_index1

        if name_index is None:
            raise ValueError('No SGD parameters found in %s' % self.root_dir)

        path_read = '%s/sgd_parameters_%d.json' % (self.root_dir, name_index)
        with open(path_read, 'r') as reader:
            deserialized_d = json.load(reader)

        assert deserialized_d['fq_name'] == self.get_class_fq_name()
        self._lr = float(deserialized_d['lr'])
        assert self.mini_batch_size == int(deserialized_d['mini_batch_size'])
        assert self.num_items == int(deserialized_d['num_items'])
        self.anneal_every_num_epochs = float(deserialized_d['anneal_every_num_epochs'])

        self._load_from_class_dict(deserialized_d)

    def _save_state(self, iteration_index):
        self._total_time_elapsed += (time.time() - self._time_mark)

        print("iteration %d: Storing model and rnd state under %s. Time: %s"
              % (iteration_index, self.root_dir, '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))
        
        model_write_path = "%s/model_%d.npy" % (self.root_dir, iteration_index)
        random_state_write_path = "%s/rnd_state_%d.npy" % (self.root_dir, iteration_index)
        np.save(model_write_path, self._x)
        with open(random_state_write_path, "w") as f:
            pickle.dump(random.getstate(), f)

        self._num_saved += 1
        self._next_save_iter = int(self._save_every * (self._num_saved + 1))

        self._time_mark = time.time()

    def _load_saved_state(self, min_iteration_index):
        """ Loads previously saved model and solver parameters.
        
        If an iteration number is specified it looks for file name that corresponds to it
        and loads. If no iteration number is specified, it finds the highest saved one.
        
        Args:
            min_iteration_index: minimum iteration to look for or None if
        Returns:
            iteration_index: iteration for which a model and random state was found, None if a minimum was requested
                but not found, 0 if no minimum was requested and nothing was found
            model: the stored model corresponding to iteration_index or None
            random_state: the random generator state corresponding to iteration_index or None
        """

        if min_iteration_index is not None:
            st = None
            for f in glob.glob(self.root_dir + "/model_*.npy"):
                file_name_no_extension = op.splitext(op.basename(f))[0]
                iteration_index = int(file_name_no_extension.split("_")[1])
                if iteration_index >= min_iteration_index and (st is None or st >= iteration_index):
                    st = iteration_index
        else:
            st = 0
            for f in glob.glob(self.root_dir + "/model_*.npy"):
                file_name_no_extension = op.splitext(op.basename(f))[0]
                iteration_index = int(file_name_no_extension.split("_")[1])
                if iteration_index > st:
                    st = iteration_index

        if st is not None and st > 0:
            model_file = "%s/model_%d.npy" % (self.root_dir, st)
            rnd_state_file = "%s/rnd_state_%d.npy" % (self.root_dir, st)
            print('Loading: %s %s' % (model_file, rnd_state_file))
            model = np.load(model_file)
            with open(rnd_state_file, "r") as f:
                random_state = pickle.load(f)
            return st, model, random_state
        else:
            return st, None, None


class SgdSolverAdam(SgdSolver):
    """
    The best all-around variant.
    It consistently performed the best in all of my applications.
    """
    def __init__(self):
        super(SgdSolverAdam, self).__init__()
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        self.m, self.v = None, None
        self.buf1, self.buf2 = None, None

    def get_class_dict(self):
        d = self._init_display_dict()
        d.update({'beta1': self.beta1, 'beta2': self.beta2, 'eps': self.eps})
        return d

    def _load_from_class_dict(self, dictionary):
        self.beta1 = dictionary['beta1']
        self.beta2 = dictionary['beta2']
        self.eps = dictionary['eps']

    def _derived_init(self):
        self.m = np.zeros_like(self._x)
        self.v = np.zeros_like(self._x)
        self.buf1 = np.empty_like(self._x)
        self.buf2 = np.empty_like(self._x)

    def _get_update_and_loss_naive(self):
        loss, grad = self._objective_func()
        # The following create large temporary objects that ruin the CPU cache. This cost is significant even relative
        # to complex models. (For instance compare SgdSolverAdam and SgdSolverMomentumNag on the same deep model.)
        # This would be far more efficient with an explicit for-loop in C (or Cython).
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1)
        v_hat = self.v / (1 - self.beta2)
        return loss, self._lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def _get_update_and_loss(self):
        loss, grad = self._objective_func()
        # Even though much faster than the naive version by avoiding large temporary objects, it now has many
        # (expensive) python function invocations.
        # This would be significantly more efficient with a few explicit for-loops in C (or Cython).
        # Following optimizations over the naive implementation lowered run-time by 15%-20% for a 2.6 million dimensions
        # model / grad vector and a complex multi-layer model.
        # in-place equivalent but much faster than: self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.m *= self.beta1
        self.m += (1 - self.beta1) * grad
        # in-place equivalent but much faster than: self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        self.v *= self.beta2
        # self.v += (1 - self.beta2) * (grad ** 2)
        np.multiply(grad, grad, out=self.buf1)
        self.v += (1 - self.beta2) * self.buf1
        # overwrite self.buf1 to hold m_hat
        np.multiply((self._lr / (1 - self.beta1)), self.m, out=self.buf1)
        m_hat = self.buf1
        v_hat = self.buf2
        np.divide(self.v, (1 - self.beta2), out=v_hat)
        # the following are safe to do in-place and they seem to lower computation cost a little
        np.sqrt(v_hat, out=v_hat)
        np.add(v_hat, self.eps, out=v_hat)
        return loss, m_hat / v_hat


class SgdSolverMomentum(SgdSolver):
    def __init__(self):
        super(SgdSolverMomentum, self).__init__()
        self.momentum_factor = 0.95

        self.cum_v = None

    def get_class_dict(self):
        d = self._init_display_dict()
        d.update({'momentum_factor': self.momentum_factor})
        return d

    def _load_from_class_dict(self, dictionary):
        self.momentum_factor = dictionary['momentum_factor']

    def set_momentum_factor(self, momentum_factor):
        self.momentum_factor = momentum_factor

    def _derived_init(self):
        self.cum_v = np.zeros_like(self._x)

    def _get_update_and_loss(self):
        loss, grad = self._objective_func()
        # in-place much faster than: self.cum_v = self.momentum_factor * self.cum_v + self._lr * grad
        self.cum_v *= self.momentum_factor
        self.cum_v += self._lr * grad
        return loss, self.cum_v


class SgdSolverMomentumNag(SgdSolver):
    """
    Nesterov accelerated gradient.
    A variant of the momentum solver that usually converges faster, often significantly faster.
    """
    def __init__(self):
        super(SgdSolverMomentumNag, self).__init__()
        self.momentum_factor = 0.95

        self.cum_v = None
        self.part = None
        self._x_sav = None

    def get_class_dict(self):
        d = self._init_display_dict()
        d.update({'momentum_factor': self.momentum_factor})
        return d

    def _load_from_class_dict(self, dictionary):
        self.momentum_factor = dictionary['momentum_factor']

    def set_momentum_factor(self, momentum_factor):
        self.momentum_factor = momentum_factor

    def _derived_init(self):
        self.cum_v = np.zeros_like(self._x)
        self.part = np.empty_like(self._x)
        self._x_sav = np.empty_like(self._x)

    def _get_update_and_loss_naive(self):
        part = self.momentum_factor * self.cum_v
        np.copyto(self._x_sav, self._x)  # save current self._x
        self._x -= self.part  # do Nesterov jump, then evaluate at new location
        loss, grad = self._objective_func()
        np.copyto(self._x, self._x_sav)  # restore current self._x
        self.cum_v = part + self._lr * grad
        return loss, self.cum_v

    def _get_update_and_loss(self):
        # Interesting impl note:
        # A seemingly correct implementation where we first do self._x -= self.part and then self._x += self.part was
        # found to have worse convergence, apparently due to numerical issues, and was not actually faster.
        # Pre-allocating and re-using self._x_sav was found to be a significant improvement in running time.
        np.multiply(self.momentum_factor, self.cum_v, out=self.part)
        np.copyto(self._x_sav, self._x)  # save current self._x
        self._x -= self.part  # do Nesterov jump, then evaluate at new location
        loss, grad = self._objective_func()
        np.copyto(self._x, self._x_sav)  # restore current self._x
        np.multiply(self._lr, grad, out=self.cum_v)
        self.cum_v += self.part
        return loss, self.cum_v


class SgdSolverStandard(SgdSolver):
    """
    Plain Old Stochastic Gradient Descent. Usually not competitive.
    """

    def __init__(self):
        super(SgdSolverStandard, self).__init__()

    def get_class_dict(self):
        return self._init_display_dict()

    def _load_from_class_dict(self, dictionary):
        pass

    def _derived_init(self):
        pass

    def _get_update_and_loss(self):
        loss, grad = self._objective_func()
        return loss, self._lr * grad

