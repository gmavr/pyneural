import time

import numpy as np

from context import pyneural

import pyneural.ce_softmax_layer as ce_sm
import pyneural.embedding_layer as em
import pyneural.gru_layer as gru
import pyneural.rnn_batch_layer as rbl
import pyneural.rnn_layer as rl


"""Samples of execution speed measurements of layer components.
For inspection and experimentation.
"""


def softmax_layer_time():
    """
    Measures CESoftmax times for batched versions against loop over the non-batched versions.
    Note: Batches are FULL. Effective performance of batched versions will be lower when some sequences are shorter
    than the batch size.
    Also verifies that loss and derivatives are the same.
    """
    dim_d, dim_k = 25, 50
    batch_size = 25  # number of sequences
    max_seq_length = 20
    num_max_seq_blocks = 50  # number of blocks, a loop for both versions

    dtype = np.float64

    asserts_on = True
    ce = ce_sm.CESoftmaxLayer(dim_k, dim_d, dtype, asserts_on=asserts_on)
    # ce = ce_sm.CESoftmaxLayerFixedLength(dim_k, dim_d, max_seq_length, dtype, asserts_on=asserts_on)
    ce_batch = ce_sm.CESoftmaxLayerBatch(dim_k, dim_d, max_seq_length, batch_size, dtype, asserts_on=asserts_on)
    ce.init_parameters_storage()

    np.random.seed(seed=47)
    ce.model_normal_init(sd=0.1)

    ce_batch.init_parameters_storage(np.copy(ce.get_model()))

    labels = np.zeros((num_max_seq_blocks * max_seq_length, batch_size), dtype=np.int)
    # each output sample has a single coordinate set
    for i in range(num_max_seq_blocks * max_seq_length):
        labels[i, 0:batch_size] = np.random.randint(0, dim_k, batch_size)

    data_t = 0.5 * np.random.standard_normal((batch_size, num_max_seq_blocks * max_seq_length, dim_d)).astype(dtype)
    data = np.empty((num_max_seq_blocks * max_seq_length, batch_size, dim_d), dtype=dtype)
    for i in range(num_max_seq_blocks * max_seq_length):
        data[i] = np.copy(data_t[:, i, :])

    loss = np.zeros(num_max_seq_blocks, dtype=dtype)
    loss_batched = np.empty(num_max_seq_blocks, dtype=dtype)

    grad_accum = np.zeros((num_max_seq_blocks, ce.get_num_p()), dtype=dtype)
    grad_batched = np.empty((num_max_seq_blocks, ce.get_num_p()), dtype=dtype)

    # (M, N, D)
    delta_err = np.empty((num_max_seq_blocks * max_seq_length, batch_size, dim_d), dtype=dtype)
    delta_err_batch = np.empty((num_max_seq_blocks * max_seq_length, batch_size, dim_d), dtype=dtype)

    print("Starting comparison of batched sequence vs loop over non-batched sequences")

    seq_lengths = np.empty((batch_size,), dtype=np.int)
    seq_lengths.fill(max_seq_length)

    start_time = time.time()
    for j in range(num_max_seq_blocks):
        loss_batched[j] = ce_batch.forward(data[(j * max_seq_length):((j + 1) * max_seq_length)],
                                           labels[(j * max_seq_length):((j + 1) * max_seq_length), :],
                                           seq_lengths)
        # ce_batch.back_propagation_batch()
        delta_err_batch[(j * max_seq_length):((j + 1) * max_seq_length), :] = ce_batch.backwards()
        grad_batched[j] = np.copy(ce_batch.get_gradient())
    time_elapsed = (time.time() - start_time)
    print("total time elapsed for %d batched invocations of %d sequences: %.4g sec" %
          (num_max_seq_blocks, batch_size, time_elapsed))

    start_time = time.time()
    for i in range(batch_size):
        for j in range(num_max_seq_blocks):
            loss[j] += ce.forward(data_t[i, (j * max_seq_length):((j + 1) * max_seq_length)],
                                  labels[(j * max_seq_length):((j + 1) * max_seq_length), i])
            # ce.back_propagation_batch()
            delta_err[(j * max_seq_length):((j + 1) * max_seq_length), i] = ce.backwards()
            grad_accum[j] += ce.get_gradient()
    time_elapsed = (time.time() - start_time)
    print("total time elapsed for %d x %d non-batched invocations of 1 sequence: %.4g sec"
          % (batch_size, num_max_seq_blocks, time_elapsed))

    rtol, atol = 1e-15, 1e-15
    assert(np.allclose(loss_batched, loss, rtol=rtol, atol=atol))
    assert(np.allclose(grad_accum, grad_batched, rtol=1e-13, atol=1e-13))
    assert(np.allclose(delta_err, delta_err_batch, rtol=rtol, atol=atol))


def embedding_time():
    num_samples = 1000
    dim_k, dim_d = 75000, 200
    dtype = np.float32

    em_obj = em.EmbeddingLayer(dim_k, dim_d, dtype)

    em_obj.init_parameters_storage()

    np.random.seed(seed=47)
    em_obj.model_normal_init(1.0)

    x = np.random.randint(0, dim_k, num_samples)
    delta_err = 0.05 * np.random.standard_normal((num_samples, dim_d)).astype(dtype)

    start_time = time.time()

    for i in range(10):
        em_obj.forward(x)
        em_obj.backwards(delta_err)

    time_elapsed = (time.time() - start_time)
    print("time elapsed: %.4g sec" % time_elapsed)


def create_random_data_em_batch(em_object, max_seq_length, batch_size):
    """
    Args:
        em_object: EmbeddingLayerBatch object
        max_seq_length: maximum sequence length (first dimension)
        batch_size: number of sequences in batch (second dimension)

    Returns:
        x: (T, B) np.int
        delta_err: (T, B, D)
        seq_lengths: (B, ) np.int
    """
    dtype = em_object.get_dtype()
    dim_k, dim_d = em_object.dim_k, em_object.dim_d

    x = np.zeros((max_seq_length, batch_size), dtype=np.int)
    delta_err = np.zeros((max_seq_length, batch_size, dim_d), dtype=dtype)

    # have few repeated items in each sequence (was needed for removed experiment)
    num_unique = min(dim_k, 10)
    assert max_seq_length >= 5
    seq_lengths = np.random.randint(max_seq_length - 5, max_seq_length, batch_size)
    for j in xrange(batch_size):
        seq_length = seq_lengths[j]
        if seq_length > 0:
            indices = np.random.randint(0, dim_k, num_unique)
            x[0:seq_length, j] = indices[np.random.randint(0, num_unique, seq_length)]
            delta_err[0:seq_length, j, :] = 0.01 * np.random.standard_normal((seq_length, dim_d)).astype(dtype)

    return x, delta_err, seq_lengths


def embedding_batched_time():
    batch_size, max_seq_length = 200, 50
    dim_k, dim_d = 5000, 50
    dtype = np.float32

    em_obj_batch = em.EmbeddingLayerBatch(dim_k, dim_d, max_seq_length, batch_size, dtype)
    em_obj_batch.init_parameters_storage()

    em_obj_batch.model_normal_init(0.1)
    x, delta_err, seq_lengths = create_random_data_em_batch(em_obj_batch, max_seq_length, batch_size)

    start_time = time.time()

    for i in range(10):
        em_obj_batch.forward(x, seq_lengths)
        em_obj_batch.backwards(delta_err)

    time_elapsed = (time.time() - start_time)
    print("time elapsed: %.4g sec" % time_elapsed)


def rnn_time():
    """
    Measures times for batched versions against loop over the non-batched versions.
    For forward pass only, it also compares batched invocation with time at 0-th and 1-st dimension (former is faster.)
    Note: Batches are full. Effective performance of batched versions will be lower with sequences of smaller lengths.
    """
    dim_d, dim_h = 25, 50
    batch_size = 20
    max_seq_length = 20
    bptt_steps = max_seq_length
    num_max_seq_blocks = 50
    dtype = np.float64

    rnn_layer = rl.RnnLayer(dim_d, dim_h, max_seq_length, dtype, bptt_steps=bptt_steps)
    rnn_batch_layer_t1 = rbl.RnnBatchLayerTime2nd(dim_d, dim_h, max_seq_length, batch_size, dtype,
                                                  bptt_steps=bptt_steps)
    rnn_batch_layer = rbl.RnnBatchLayer(dim_d, dim_h, max_seq_length, batch_size, dtype, bptt_steps=bptt_steps)

    model = 0.1 * np.random.standard_normal((rnn_layer.get_num_p(),)).astype(dtype)
    hs_init = 0.01 * np.random.standard_normal((batch_size, dim_h)).astype(dtype)

    rnn_layer.init_parameters_storage(model)
    rnn_batch_layer_t1.init_parameters_storage(model)
    rnn_batch_layer.init_parameters_storage(model)

    data_t = 0.5 * np.random.standard_normal((batch_size, num_max_seq_blocks * max_seq_length, dim_d)).astype(dtype)
    data = np.empty((num_max_seq_blocks * max_seq_length, batch_size, dim_d), dtype=dtype)
    for i in range(num_max_seq_blocks * max_seq_length):
        data[i] = np.copy(data_t[:, i, :])

    start_time = time.time()
    for i in range(batch_size):
        rnn_layer.set_init_h(hs_init[i])
        for j in range(num_max_seq_blocks):
            rnn_layer.forward(data_t[i, (j * max_seq_length):((j + 1) * max_seq_length)])
    time_elapsed = (time.time() - start_time)
    print("total time elapsed for %d x %d non-batched invocations of 1 sequence: %.4g sec"
          % (batch_size, num_max_seq_blocks, time_elapsed))

    seq_lengths = np.empty((batch_size, ), dtype=np.int)
    seq_lengths.fill(max_seq_length)

    start_time = time.time()
    rnn_batch_layer_t1.set_init_h(hs_init)
    rnn_batch_layer.set_init_h(hs_init)
    for j in range(num_max_seq_blocks):
        rnn_batch_layer_t1.forward(data_t[:, (j * max_seq_length):((j + 1) * max_seq_length)], seq_lengths)
    time_elapsed = (time.time() - start_time)
    print("total time elapsed for %d batched invocations of %d sequences: %.4g sec" %
          (num_max_seq_blocks, batch_size, time_elapsed))

    start_time = time.time()
    rnn_batch_layer.set_init_h(hs_init)
    for j in range(num_max_seq_blocks):
        rnn_batch_layer.forward(data[(j * max_seq_length):((j + 1) * max_seq_length)], seq_lengths)
    time_elapsed = (time.time() - start_time)
    print("total time elapsed for %d batched invocations of %d sequences: %.4g sec" %
          (num_max_seq_blocks, batch_size, time_elapsed))

    # forward-backward

    delta_upper1 = 0.1 * np.random.standard_normal((batch_size, num_max_seq_blocks * max_seq_length, dim_h))\
        .astype(dtype)
    delta_upper2 = np.empty((num_max_seq_blocks * max_seq_length, batch_size, dim_h), dtype=dtype)
    for i in range(num_max_seq_blocks * max_seq_length):
        delta_upper2[i] = np.copy(delta_upper1[:, i, :])

    start_time = time.time()
    for i in range(batch_size):
        rnn_layer.set_init_h(hs_init[i])
        for j in range(num_max_seq_blocks):
            rnn_layer.forward(data_t[i, (j * max_seq_length):((j + 1) * max_seq_length)])
            rnn_layer.backwards(delta_upper1[i, (j * max_seq_length):((j + 1) * max_seq_length)])
    time_elapsed = (time.time() - start_time)
    print("total time elapsed for %d x %d non-batched invocations of 1 sequence: %.4g sec"
          % (batch_size, num_max_seq_blocks, time_elapsed))

    start_time = time.time()
    rnn_batch_layer.set_init_h(hs_init)
    for j in range(num_max_seq_blocks):
        rnn_batch_layer.forward(data[(j * max_seq_length):((j + 1) * max_seq_length)], seq_lengths)
        rnn_batch_layer.backwards(delta_upper2[(j * max_seq_length):((j + 1) * max_seq_length)])
    time_elapsed = (time.time() - start_time)
    print("total time elapsed for %d batched invocations of %d sequences: %.4g sec" %
          (num_max_seq_blocks, batch_size, time_elapsed))

    # batch size, backprop speedup |  (dim_d, dim_h = 25, 50  max_seq_length = 20 num_max_seq_blocks = 50)
    # 2 1.1 | 5, 1.8 | 10, 2.5 | 20, 3.5 | 50, 4.5 | 100, 4.8 | 200 5.3


def create_gru_random_data(dim_x, dim_y, dtype, num_params, num_samples):
    model = 0.1 * np.random.standard_normal(num_params).astype(dtype)
    x = 0.5 * np.random.standard_normal((num_samples, dim_x)).astype(dtype)
    y = 0.5 * np.random.standard_normal((num_samples, dim_y)).astype(dtype)
    h_init = 0.1 * np.random.standard_normal(dim_y).astype(dtype)
    return x, y, model, h_init


def gru_forward_time():
    num_samples = 100
    dim_d, dim_h, max_seq_length = 500, 200, num_samples
    dtype = np.float64

    gru1 = gru.GruLayer(dim_d, dim_h, max_seq_length, dtype)
    gru2 = gru.GruLayer(dim_d, dim_h, max_seq_length, dtype)

    np.random.seed(seed=47)
    x, _, model, _ = create_gru_random_data(dim_d, dim_h, dtype, gru1.get_num_p(), num_samples)

    gru1.init_parameters_storage(model=model)
    gru2.init_parameters_storage(model=np.copy(model))

    # Run them once before the loop to warm up cache. Whichever version runs first usually takes longer to execute.
    # The timed ones are run interleaved.
    # For dim_d = 500, dim_h = 200, n = 100 the optimized version is faster by 2%-8%.

    y1 = gru1.forward(x)
    y2 = gru2.forward_batch_debug(x)

    num_iter = 5
    times1 = []
    times2 = []

    for i in range(num_iter):
        start_time = time.time()
        gru1.forward(x)
        times1.append(time.time() - start_time)

        start_time = time.time()
        gru2.forward_batch_debug(x)
        times2.append(time.time() - start_time)

    t1 = sum(times1)
    print("time elapsed (opt): %.4g sec" % t1)
    t2 = sum(times2)
    print("time elapsed (dbg): %.4g sec" % t2)

    assert(np.allclose(y1, y2, rtol=1e-14, atol=1e-14))


if __name__ == "__main__":
    softmax_layer_time()
    embedding_time()
    embedding_batched_time()
    rnn_time()
    gru_forward_time()
