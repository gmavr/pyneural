cimport cython
from cython cimport view
import numpy as np
cimport numpy as np

from pyneural.ctypes cimport np_floats_t, np_length_ints_t, np_index_ints_t

"""
Cython implementation of components of embedding layers providing substantial speed up over python-only implementation.
Implementation here is specific to how it is invoked by the relevant python code (e.g. memory views are often contiguous
rather than more general layouts.)
"""

def forward(np.ndarray[np_index_ints_t, ndim=1] x not None,
            np.ndarray[np_floats_t, ndim=2] embeddings not None,
            np.ndarray[np_floats_t, ndim=2] y not None):
    cdef np_index_ints_t[:] x_view = x
    cdef np_floats_t[:, ::1] embeddings_view = embeddings
    cdef np_floats_t[:, ::1] y_view = y
    forward_c(x_view, embeddings_view, y_view)


@cython.boundscheck(False)
cdef inline void forward_c(np_index_ints_t[:] x_view,
                           np_floats_t[:, ::1] embeddings_view,
                           np_floats_t[:, ::1] y_view) nogil:
    cdef Py_ssize_t i, d
    for i in range(x_view.shape[0]):
        for d in range(embeddings_view.shape[1]):
            y_view[i, d] = embeddings_view[x_view[i], d]


def backwards_zero_grad(np.ndarray[np_index_ints_t, ndim=1] x not None,
                        np.ndarray[np_floats_t, ndim=2] grad not None):
    cdef np_index_ints_t[:] x_view = x
    cdef np_floats_t[:, ::1] grad_view = grad
    backwards_zero_grad_c(x_view, grad_view)


@cython.boundscheck(False)
cdef inline void backwards_zero_grad_c(np_index_ints_t[:] x_view,
                                       np_floats_t[:, ::1] grad_view) nogil:
    cdef Py_ssize_t i, k
    for i in range(x_view.shape[0]):
        for k in range(grad_view.shape[1]):
            grad_view[x_view[i], k] = 0.0


def backwards(np.ndarray[np_index_ints_t, ndim=1] x not None,
              np.ndarray[np_floats_t, ndim=2] delta_err not None,
              np.ndarray[np_floats_t, ndim=2] grad not None):
    cdef np_index_ints_t[:] x_view = x
    cdef np_floats_t[:, :] delta_err_view = delta_err
    cdef np_floats_t[:, ::1] grad_view = grad
    backwards_c(x_view, delta_err_view, grad_view)


@cython.boundscheck(False)
cdef inline void backwards_c(np_index_ints_t[:] x_view,
                             np_floats_t[:, :] delta_err_view,
                             np_floats_t[:, ::1] grad_view) nogil:
    cdef Py_ssize_t i, k
    for i in range(x_view.shape[0]):
        for k in range(grad_view.shape[1]):
            grad_view[x_view[i], k] += delta_err_view[i, k]


def forward_batch(np.ndarray[np_index_ints_t, ndim=2] x not None,
                  np.ndarray[np_floats_t, ndim=2] embeddings not None,
                  np.ndarray[np_floats_t, ndim=3] y not None,
                  np.ndarray[np_length_ints_t, ndim=1] seq_lengths not None):
    cdef np_index_ints_t[:, :] x_view = x
    cdef np_floats_t [:, ::1] embeddings_view = embeddings
    cdef np_floats_t [:, :, ::view.contiguous] y_view = y
    cdef np_length_ints_t [::1] seq_lengths_view = seq_lengths
    forward_batch_c(x_view, embeddings_view, y_view, seq_lengths_view)


@cython.boundscheck(False)
cdef void forward_batch_c(np_index_ints_t[:, :] x_view,
                          np_floats_t[:, ::1] embeddings_view,
                          np_floats_t[:, :, ::view.contiguous] y_view,
                          np_length_ints_t[::1] seq_lengths_view) nogil:
    cdef Py_ssize_t i, j, k
    cdef np_index_ints_t em_index

    for j in range(x_view.shape[1]):
        for i in range(seq_lengths_view[j]):
            em_index = x_view[i, j]
            for d in range(embeddings_view.shape[1]):
                y_view[i, j, d] = embeddings_view[em_index, d]
        # we must zero out beyond sequence length per contract
        for i in range(seq_lengths_view[j], y_view.shape[0]):
            for d in range(embeddings_view.shape[1]):
                y_view[i, j, d] = 0.0


def backwards_batch(np.ndarray[np_index_ints_t, ndim=2] x not None,
                    np.ndarray[np_floats_t, ndim=3] delta_err not None,
                    np.ndarray[np_length_ints_t, ndim=1] seq_lengths not None,
                    np.ndarray[np_floats_t, ndim=2] grad not None):
    cdef np_index_ints_t[:, :] x_view = x
    cdef np_floats_t[:, :, ::view.contiguous] delta_err_view = delta_err
    cdef np_length_ints_t[::1] seq_lengths_view = seq_lengths
    cdef np_floats_t[:, ::1] grad_view = grad
    backwards_batch_c(x_view, delta_err_view, seq_lengths_view, grad_view)


@cython.boundscheck(False)
cdef void backwards_batch_c(np_index_ints_t[:, :] x_view,
                            np_floats_t[:, :, ::view.contiguous] delta_err_view,
                            np_length_ints_t[::1] seq_lengths_view,
                            np_floats_t[:, ::1] grad_view) nogil:
    cdef Py_ssize_t i, j, k
    for j in range(x_view.shape[1]):
        for i in range(seq_lengths_view[j]):
            for k in range(grad_view.shape[1]):
                grad_view[x_view[i, j], k] += delta_err_view[i, j, k]


def backwards_batch_zero_grad(np.ndarray[np_index_ints_t, ndim=2] x not None,
                              np.ndarray[np_length_ints_t, ndim=1] seq_lengths,
                              np.ndarray[np_floats_t, ndim=2] grad not None):
    cdef np_index_ints_t[:, :] x_view = x
    cdef np_length_ints_t[::1] seq_lengths_view = seq_lengths
    cdef np_floats_t[:, ::1] grad_view = grad
    backwards_batch_zero_grad_c(x_view, seq_lengths_view, grad_view)


# for unknown reasons declaring np_index_ints_t[:, :] instead of np_index_ints_t[:, ::1]
# (when in fact we know that ::1 suffices) results in faster code (mac os, clang)
@cython.boundscheck(False)
cdef void backwards_batch_zero_grad_c(np_index_ints_t[:, :] x_view,
                                      np_length_ints_t[::1] seq_lengths_view,
                                      np_floats_t[:, ::1] grad_view) nogil:
    cdef Py_ssize_t i, j, k
    for j in range(x_view.shape[1]):
        for i in range(seq_lengths_view[j]):
            for k in range(grad_view.shape[1]):
                grad_view[x_view[i, j], k] = 0.0
