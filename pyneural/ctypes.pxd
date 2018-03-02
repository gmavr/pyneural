cimport cython
import numpy as np
cimport numpy as np  # numpy.pxd

# The number of specializations of a function produced at compile time is the cartesian product of the subtypes of all
# fused types present in the function signature. Therefore we should try to limit the number of allowed types as it
# blows-up the binary code size and compilation time.

ctypedef fused np_floats_t:
    np.float32_t
    np.float64_t

# Array indexing type.
# As a trade-off between compactness of data representation and compiled code size explosion allow some small types but
# only their unsigned versions.
# Interestingly, on a 2012 intel core i7 using clang, indexing by the native index type np.intp (usually np.int64) may
# be only marginally faster if any faster at all. So if memory matters, use smaller indexing types without reservations.
ctypedef fused np_index_ints_t:
    np.uint8_t
    # np.uint16_t  # consider adding back
    np.int32_t
    np.int64_t

# Array length type.
# Allow unsigned 8bit for enabling memory compactness when possible.
ctypedef fused np_length_ints_t:
    np.uint8_t
    np.int32_t
    np.int64_t
