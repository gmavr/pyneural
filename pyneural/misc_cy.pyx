cimport cython
from cython cimport view
import numpy as np
cimport numpy as np

from libc.math cimport exp as c_exp
from libc.math cimport log as c_log

from pyneural.ctypes cimport np_floats_t, np_index_ints_t, np_length_ints_t

"""
Cython implementation of miscellaneous components providing substantial speed up over python-only implementation.
Implementation here is specific to how it is invoked by the relevant python code (e.g. memory views are often contiguous
rather than more general layouts, because we know how the python code invokes these)
"""

# Note on exponentiation function:
# C math exp or numpy.exp() are remarkably expensive to compute functions. numpy.tanh() is much cheaper.
# Unfortunately exp is used heavily standard softmax, GRU, CRF, and these cannot be sped up with Cython.


# Used by neural_base.py
def validate_zero_padding_3d(np.ndarray[np_floats_t, ndim=3] array not None,
                             int max_seq_length, int max_num_sequences,
                             np.ndarray[np_length_ints_t, ndim=1] seq_lengths not None):
    cdef np_floats_t[:, :, ::view.contiguous] array_view = array
    cdef np_length_ints_t[::1] seq_lengths_view = seq_lengths
    cdef int ret = validate_zero_padding_3d_c(array_view, max_seq_length, max_num_sequences, seq_lengths_view)
    return ret

@cython.boundscheck(False)
cdef int validate_zero_padding_3d_c(np_floats_t[:, :, ::view.contiguous] array_view,
                                    int max_seq_length, int max_num_sequences,
                                    np_length_ints_t[::1] seq_lengths_view) nogil:
    cdef Py_ssize_t i, j, k
    for j in range(max_num_sequences):
        if seq_lengths_view[j] < max_seq_length:
            for i in range(seq_lengths_view[j], max_seq_length):
                for k in range(array_view.shape[2]):
                    if array_view[i, j, k] != 0.0:
                        return 0
    return 1


def validate_zero_padding_2d_int(np.ndarray[np_index_ints_t, ndim=2] array not None,
                                 int max_seq_length, int max_num_sequences,
                                 np.ndarray[np_length_ints_t, ndim=1] seq_lengths not None):
    cdef np_index_ints_t[:, ::1] array_view = array
    cdef np_length_ints_t[::1] seq_lengths_view = seq_lengths
    cdef int ret = validate_zero_padding_2d_int_c(array_view, max_seq_length, max_num_sequences, seq_lengths_view)
    return ret

@cython.boundscheck(False)
cdef int validate_zero_padding_2d_int_c(np_index_ints_t[:, ::1] array_view,
                                        int max_seq_length, int max_num_sequences,
                                        np_length_ints_t[::1] seq_lengths_view) nogil:
    cdef Py_ssize_t i, j, k
    for j in range(max_num_sequences):
        if seq_lengths_view[j] < max_seq_length:
            for i in range(seq_lengths_view[j], max_seq_length):
                if array_view[i, j] != 0:
                    return 0
    return 1


# Used by GruBatchLayer
def add_to_diag_batch(np.ndarray[np_floats_t, ndim=4] a not None,
                      np.ndarray[np_floats_t, ndim=3] b not None):
    # we allocate in python immediately before this call, therefore we can guarantee contiguous
    cdef np_floats_t[:, :, :, ::1] a_view = a
    cdef np_floats_t[:, :, ::view.contiguous] b_view = b
    add_to_diag_batch_c(a_view, b_view)

@cython.boundscheck(False)
cdef inline void add_to_diag_batch_c(
        np_floats_t[:, :, :, ::1] a_view, np_floats_t[:, :, ::view.contiguous] b_view) nogil:
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t n = b_view.shape[0]
    cdef Py_ssize_t b = b_view.shape[1]
    cdef Py_ssize_t h = b_view.shape[2]
    for i in range(n):
        for j in range(b):
            for k in range(h):
                a_view[i, j, k, k] += b_view[i, j, k]


# Following could have benn used by CRF, but it is faster only in a narrow case.

# log_sum_exp_opt in cython is slightly slower than pure python on mac os for both float32 and float64 and on linux
# for float32. On linux it is moderately faster for float64. This is with full cython-ization of all calls.
# Using np.exp or C math exp made no difference.
# If function invokes numpy calls, it is even slower, much slower if all are numpy calls.
# Conclusion: Not worth using cython here. Cython can be slower when it spans over or replaces many numpy calls and
# there are no loops with very little work inside them. This is surprising but seen consistently.

def log_sum_exp_opt(np.ndarray[np_floats_t, ndim=2] s_array,
                    np_floats_t scale_term,
                    int axis,
                    np.ndarray[np_floats_t, ndim=2] buf_k2,
                    np.ndarray[np_floats_t, ndim=1] out_rs_vector,
                    np.ndarray[np_floats_t, ndim=1] out_s_vector):

    cdef np_floats_t[:, ::1] s_array_view = s_array
    cdef np_floats_t[:, ::1] buf_k2_view = buf_k2
    mysubtract(s_array_view, scale_term, buf_k2_view)

    # np.exp(buf_k2, out=buf_k2)
    myexp(buf_k2_view)

    cdef np_floats_t[::1] out_rs_vector_view = out_rs_vector

    # np.sum(buf_k2, axis=axis, out=out_rs_vector)
    mysum(buf_k2_view, axis, out_rs_vector_view)

    cdef np_floats_t[::1] out_s_vector_view = out_s_vector

    # np.log(out_rs_vector, out=out_rs_vector)
    mylog(out_rs_vector_view)

    myadd(out_rs_vector_view, scale_term, out_s_vector_view)

    return scale_term

@cython.boundscheck(False)
cdef inline void mysum(np_floats_t[:, ::1] x_view, int axis, np_floats_t[::1] out_view) nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t m = x_view.shape[0]
    cdef Py_ssize_t n = x_view.shape[1]
    if axis == 0:
        for j in range(0, n):
            out_view[j] = x_view[0, j]
        for i in range(1, m):
            for j in range(0, n):
                out_view[j] += x_view[i, j]
    else:
        for i in range(0, m):
            out_view[i] = x_view[i, 0]
            for j in range(1, n):
                out_view[i] += x_view[i, j]

@cython.boundscheck(False)
cdef inline void mysubtract(np_floats_t[:, ::1] x_view, np_floats_t term, np_floats_t[:, ::1] out_view) nogil:
    cdef Py_ssize_t i, j
    for i in range(x_view.shape[0]):
        for j in range(x_view.shape[1]):
            out_view[i,j] = x_view[i,j] - term

@cython.boundscheck(False)
cdef inline void myexp(np_floats_t[:, ::1] x_view) nogil:
    cdef Py_ssize_t i, j
    for i in range(x_view.shape[0]):
        for j in range(x_view.shape[1]):
            x_view[i,j] = c_exp(x_view[i,j])

@cython.boundscheck(False)
cdef inline void mylog(np_floats_t[::1] x_view) nogil:
    cdef Py_ssize_t i
    for i in range(x_view.shape[0]):
        x_view[i] = c_log(x_view[i])

@cython.boundscheck(False)
cdef inline void myadd(np_floats_t[::1] x_view, np_floats_t term, np_floats_t[::1] out_view) nogil:
    cdef Py_ssize_t i
    for i in range(x_view.shape[0]):
        out_view[i] = x_view[i] + term
