import time

import numpy as np

import bidir_rnn_layer_test as br_test
import dataset as dst
import embedding_layer as em
import layers
import sgd
from layers_test import create_rnn_random_data

""" Samples of model fitting.

Meant for experimentation, measurement of effect on computation time of changes, demonstration and basic validation
that training loss indeed decreases etc.
"""


def run_rnn_sgd_with_boundaries():
    batch_size = 10
    num_samples = 20
    dim_d, dim_h, dim_k = (20, 80, 10)
    dtype = np.float32

    rnn_obj = layers.RnnSoftMax((dim_d, dim_h, dim_k), batch_size, batch_size, dtype)

    print("number parameters: %d" % rnn_obj.get_num_p())

    np.random.seed(seed=47)
    model, data, labels, h_init = create_rnn_random_data(rnn_obj, num_samples)
    rnn_obj.set_init_h(h_init)
    rnn_obj.init_parameters_storage(model)

    docs_end = np.array([4, 9, 19])
    dataset = dst.InMemoryDataSetWithBoundaries(data, labels, docs_end)
    rnn_and_data = dst.LossNNAndDataSetWithBoundary(rnn_obj, dataset, batch_size)

    # The loss without the boundaries (obtained by below uncommented) appears to be the same on the average (over many
    # runs) as with the boundaries. This is evidence (but not proof) of no bug in forward and derivative of the case
    # with boundaries.
    # dataset = dt.InMemoryDataSet(data, labels)
    # rnn_and_data = rnn.LossNNAndDataSet(rnn_obj, dataset, batch_size)

    solver = sgd.SgdSolverMomentumNag()
    solver.set_momentum_factor(0.95)
    solver.set_lr(0.5)

    solver.num_epochs = 40
    solver.num_items = num_samples
    solver.mini_batch_size = batch_size
    solver.anneal_every_num_epochs = 20
    solver.report_every_num_epochs = 1

    # baseline is uniform at random predictions (i.e. all with  equal probability)
    print("baseline_loss: %f" % (np.log(dim_k)))

    solver.set_x(model)

    start_time = time.time()
    solver.sgd(rnn_and_data.forward_backward_batch)
    time_elapsed = (time.time() - start_time)
    num_iterations = solver.get_num_iterations()
    print("per iteration time: %.4g sec, per sample time: %.7e sec"
          % (time_elapsed / num_iterations, time_elapsed / (num_iterations*batch_size)))
    print("epochs=%d" % dataset.num_epochs)

    assert np.alltrue(np.equal(model, rnn_obj.get_model()))
    assert np.shares_memory(model, rnn_obj.get_model())


def run_bi_rnn_sgd():
    batch_size = 50
    num_samples = 2000
    dim_d, dim_h, dim_k = (50, 100, 20)
    dtype = np.float32

    # gru cell needs higher learning rate
    bi_rnn_obj = layers.BidirRnnSoftMax((dim_d, dim_h, dim_k), batch_size, batch_size, dtype, cell_type="basic")

    print("class: %s number parameters: %d" % (type(bi_rnn_obj).__name__, bi_rnn_obj.get_num_p()))

    np.random.seed(seed=47)
    model, data, labels, h_init = br_test.create_random_data_dense_inputs(bi_rnn_obj, num_samples)
    bi_rnn_obj.set_init_h(h_init)
    bi_rnn_obj.init_parameters_storage(model)

    dataset = dst.InMemoryDataSet(data, labels)
    rnn_and_data = dst.LossNNAndDataSet(bi_rnn_obj, dataset, batch_size)

    solver = sgd.SgdSolverMomentumNag()
    solver.set_momentum_factor(0.95)
    solver.set_lr(0.01)
    solver.num_epochs = 20
    solver.num_items = num_samples
    solver.mini_batch_size = batch_size
    solver.anneal_every_num_epochs = 5000
    solver.report_every_num_epochs = 2

    # baseline is uniform at random predictions (i.e. all with  equal probability)
    print("baseline_loss: %f" % np.log(dim_k))

    solver.set_x(model)

    start_time = time.time()
    solver.sgd(rnn_and_data.forward_backward_batch)
    time_elapsed = (time.time() - start_time)
    num_iterations = solver.get_num_iterations()
    print("per iteration time: %.4g sec, per sample time: %.7e sec"
          % (time_elapsed / num_iterations, time_elapsed / (num_iterations*batch_size)))
    print("epochs=%d" % dataset.num_epochs)

    assert np.alltrue(np.equal(model, bi_rnn_obj.get_model()))
    assert np.shares_memory(model, bi_rnn_obj.get_model())


# used by run_manual_init_sgd_long()

prng_modulus = 1024 * 1024 * 1024 * 4  # 2^32


def lcg(seed=None):
    a = 1664525
    c = 1013904223
    if seed is not None:
        lcg.previous = seed
    random_number = (lcg.previous * a + c) % prng_modulus
    lcg.previous = random_number
    return random_number

lcg.previous = 2222


def run_manual_init_sgd_long():
    """ Data set and model are initialized with our own pseudo-random number generator.

    This is so that we can replicate exactly initialization here and my C++ framework and validate results.
    """
    num_samples = 1000
    batch_size = 100
    dim_d, dim_h, dim_k = (100, 150, 10)
    dtype = np.float64

    loss_obj = layers.RnnSoftMax((dim_d, dim_h, dim_k), batch_size, batch_size, dtype, cell_type="basic",
                                 activation="tanh", asserts_on=True, grad_clip_thres=False)
    loss_obj.init_parameters_storage()

    x = np.empty((num_samples, dim_d), dtype)
    for i in xrange(num_samples):
        for j in xrange(dim_d):
            x[i, j] = float(lcg()) / float(prng_modulus) - 0.5
    # print(x)

    y_true = np.empty((num_samples, ), np.int)
    for i in xrange(num_samples):
        y_true[i] = int(dim_k * (float(lcg()) / float(prng_modulus)))
    # print(y_true)

    model = loss_obj.get_model()
    loss_obj.model_glorot_init()
    # model.fill(0.1)

    dataset = dst.InMemoryDataSet(x, y_true)
    rnn_and_data = dst.LossNNAndDataSet(loss_obj, dataset, batch_size)

    solver = sgd.SgdSolverAdam()
    solver.set_lr(0.01)
    solver.num_epochs = 50.0
    solver.num_items = num_samples
    solver.mini_batch_size = batch_size
    solver.anneal_every_num_epochs = 5000
    solver.report_every_num_epochs = 5.0

    # baseline is uniform at random predictions (i.e. all with  equal probability)
    print("baseline_loss: %f" % np.log(dim_k))

    solver.set_x(model)

    start_time = time.time()
    solver.sgd(rnn_and_data.forward_backward_batch)
    time_elapsed = (time.time() - start_time)
    num_iterations = solver.get_num_iterations()
    print("per iteration time: %.4g sec, per sample time: %.7e sec"
          % (time_elapsed / num_iterations, time_elapsed / (num_iterations*batch_size)))
    print("epochs=%d" % dataset.num_epochs)


def run_rnn_sgd_long():
    dtype = np.float32
    num_samples = 2000
    bttp_steps = batch_size = 500
    dim_d, dim_h, dim_k = 100, 200, 10
    # following on a 2.3GHz core i7 2012 mac os laptop.
    # dtype = np.float32, batch_size = 500, cell_type="basic", ADAM solver
    # Increasing input batch_size from 100 to 500 produces some benefit in execution time.
    # For the same number of params, gru is 6.5-9.5 times slower than basic rnn (ratio increases with dim_h)
    # dim_d, dim_h, dim_k = (100, 100, 10)  # gru: num_p = 61310, 1.92e-04 sec, basic: num_p = 21110, 1.715e-05
    # dim_d, dim_h, dim_k = (100, 200, 10)  # gru: num_p = 182610, 4.28e-04, basic: num_p = 62210, 2.75e-05 sec
    # dim_d, dim_h, dim_k = (100, 400, 10)  # gru: num_p = 605210, 1.60e-03 sec, basic: num_p = 204410, 6.79e-05 sec
    # dim_d, dim_h, dim_k = (100, 800, 10)  # gru: num_p = 2170410, 1.01e-02 sec basic: num_p = 728810, 1.61e-04 sec

    loss_obj = layers.RnnSoftMax((dim_d, dim_h, dim_k), batch_size, bttp_steps, dtype, cell_type="basic",
                                 activation="tanh", asserts_on=True, grad_clip_thres=False)
    loss_obj.init_parameters_storage()

    print("class: %s\t number parameters: %d" % (type(loss_obj).__name__, loss_obj.get_num_p()))

    np.random.seed(seed=47)
    model, data, labels, h_init = create_rnn_random_data(loss_obj, num_samples)
    loss_obj.set_init_h(h_init)

    loss_obj.model_glorot_init()
    # loss_obj.model_identity_glorot_init(0.1)
    model = loss_obj.get_model()

    dataset = dst.InMemoryDataSet(data, labels)
    rnn_and_data = dst.LossNNAndDataSet(loss_obj, dataset, batch_size)

    solver = sgd.SgdSolverAdam()
    solver.set_lr(0.005)
    solver.num_epochs = 20.0
    solver.num_items = num_samples
    solver.mini_batch_size = batch_size
    solver.report_every_num_epochs = 5.0

    # baseline is uniform at random predictions (i.e. all with  equal probability)
    print("baseline_loss: %f" % np.log(dim_k))

    solver.set_x(model)

    start_time = time.time()
    solver.sgd(rnn_and_data.forward_backward_batch)
    time_elapsed = (time.time() - start_time)
    num_iterations = solver.get_num_iterations()
    print("per iteration time: %.4g sec, per sample time: %.4e sec"
          % (time_elapsed / num_iterations, time_elapsed / (num_iterations*batch_size)))
    print("epochs=%d" % dataset.num_epochs)

    assert np.alltrue(np.equal(model, loss_obj.get_model()))
    assert np.shares_memory(model, loss_obj.get_model())


def run_rnn_em_sgd_long():
    batch_size = 50
    num_samples = 5000
    dim_d, dim_h, dim_k = 100, 400, 300
    dim_v = 100
    dtype = np.float32

    # in this case sigmoid activation works much better than tanh
    # gru even with fewer parameters than basic works much better
    rnn_obj = layers.RnnSoftMax((dim_d, dim_h, dim_k), batch_size, batch_size, dtype,
                                cell_type="basic", activation="sigmoid")
    em_obj = em.EmbeddingLayer(dim_k, dim_d, dtype)
    rnn_obj_em = layers.RnnEmbeddingsSoftMax(rnn_obj, em_obj)

    print("class: %s\t number parameters: %d" % (type(rnn_obj_em).__name__, rnn_obj.get_num_p()))

    np.random.seed(seed=47)
    model, data, labels, h_init = create_rnn_random_data(rnn_obj_em, num_samples, dim_v)
    rnn_obj_em.init_parameters_storage(model)
    rnn_obj.set_init_h(h_init)

    dataset = dst.InMemoryDataSet(data, labels)
    rnn_and_data = dst.LossNNAndDataSet(rnn_obj_em, dataset, batch_size)

    solver = sgd.SgdSolverAdam()
    solver.set_lr(0.005)
    solver.num_epochs = 10
    solver.num_items = num_samples
    solver.mini_batch_size = batch_size
    solver.report_every_num_epochs = 0.5

    # baseline is uniform at random predictions (i.e. all with  equal probability)
    print("baseline_loss: %f" % np.log(dim_k))

    solver.set_x(model)

    start_time = time.time()
    solver.sgd(rnn_and_data.forward_backward_batch)
    time_elapsed = (time.time() - start_time)
    num_iterations = solver.get_num_iterations()
    print("per iteration time: %.4g sec, per sample time: %.7e sec"
          % (time_elapsed / num_iterations, time_elapsed / (num_iterations*batch_size)))
    print("epochs=%d" % dataset.num_epochs)

    assert np.alltrue(np.equal(model, rnn_obj_em.get_model()))
    assert np.shares_memory(model, rnn_obj_em.get_model())


def run_rnn_class_sm_sgd_long():
    batch_size = 50
    num_samples = 10000
    dim_c = 80
    dim_d, dim_h, dim_k = (100, 400, dim_c * dim_c)

    dtype = np.float32

    word_class_mapper = layers.WordClassMapper(dim_c, dim_k)

    rnn_obj = layers.RnnClassSoftMax((dim_d, dim_h, dim_k), word_class_mapper,
                                     batch_size=batch_size, bptt_steps=batch_size, dtype=dtype)

    print("class: %s number parameters: %d" % (type(rnn_obj).__name__, rnn_obj.get_num_p()))

    np.random.seed(seed=47)
    model, data, labels, h_init = create_rnn_random_data(rnn_obj, num_samples)
    rnn_obj.set_init_h(h_init)
    rnn_obj.init_parameters_storage(model)

    dataset = dst.InMemoryDataSet(data, labels)
    rnn_and_data = dst.LossNNAndDataSet(rnn_obj, dataset, batch_size)

    solver = sgd.SgdSolverMomentumNag()
    solver.set_momentum_factor(0.95)
    solver.set_lr(0.2)

    solver.num_epochs = 10
    solver.num_items = num_samples
    solver.mini_batch_size = batch_size
    solver.report_every_num_epochs = 0.5

    # baseline is uniform at random predictions (i.e. all with  equal probability)
    print("baseline_loss: %f" % (np.log(dim_k)))
    print("baseline_loss: %f" % (np.log(2*dim_c)))

    solver.set_x(model)

    start_time = time.time()
    solver.sgd(rnn_and_data.forward_backward_batch)
    time_elapsed = (time.time() - start_time)
    num_iterations = solver.get_num_iterations()
    print("per iteration time: %.4g sec, per sample time: %.7e sec"
          % (time_elapsed / num_iterations, time_elapsed / (num_iterations*batch_size)))
    print("epochs=%d" % dataset.num_epochs)

    assert np.alltrue(np.equal(model, rnn_obj.get_model()))


if __name__ == "__main__":
    # run_rnn_sgd_with_boundaries()
    # run_bi_rnn_sgd()
    # run_manual_init_sgd_long()
    run_rnn_sgd_long()
    # run_rnn_em_sgd_long()
    # run_rnn_class_sm_sgd_long()

