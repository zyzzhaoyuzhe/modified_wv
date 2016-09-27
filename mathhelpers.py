from __future__ import with_statement

import logging
import math

from gensim import utils

import numpy as np
import scipy.sparse
from scipy.stats import entropy
import scipy.linalg
from scipy.linalg.lapack import get_lapack_funcs

from six import iteritems, itervalues, string_types
from six.moves import xrange, zip as izip

try:
    from scipy.linalg.basic import triu
except ImportError:
    from scipy.linalg.special_matrices import triu

try:
    from np import triu_indices
except ImportError:
    # np < 1.4
    def triu_indices(n, k=0):
        m = np.ones((n, n), int)
        a = triu(m, k)
        return np.where(a != 0)

blas = lambda name, ndarray: scipy.linalg.get_blas_funcs((name,), (ndarray,))[0]


def veclen(vec):
    if len(vec) == 0:
        return 0.0
    length = 1.0 * math.sqrt(sum(val**2 for _, val in vec))
    assert length > 0.0, "sparse documents must not contain any explicit zero entries"
    return length


def ret_normalized_vec(vec, length):
    if length != 1.0:
        return [(termid, val / length) for termid, val in vec]
    else:
        return list(vec)

blas_nrm2 = blas('nrm2', np.array([], dtype=float))
blas_scal = blas('scal', np.array([], dtype=float))

def unitvec(vec, norm='l2'):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.

    Output will be in the same format as input (i.e., gensim vector=>gensim vector,
    or np array=>np array, scipy.sparse=>scipy.sparse).
    """
    if norm not in ('l1', 'l2'):
        raise ValueError("'%s' is not a supported norm. Currently supported norms are 'l1' and 'l2'." % norm)
    if scipy.sparse.issparse(vec):
        vec = vec.tocsr()
        if norm == 'l1':
            veclen = np.sum(np.abs(vec.data))
        if norm == 'l2':
            veclen = np.sqrt(np.sum(vec.data ** 2))
        if veclen > 0.0:
            return vec / veclen
        else:
            return vec

    if isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=float)
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        if norm == 'l2':
            veclen = blas_nrm2(vec)
        if veclen > 0.0:
            return blas_scal(1.0 / veclen, vec)
        else:
            return vec

    try:
        first = next(iter(vec))     # is there at least one element?
    except:
        return vec

    if isinstance(first, (tuple, list)) and len(first) == 2: # gensim sparse format
        if norm == 'l1':
            length = float(sum(abs(val) for _, val in vec))
        if norm == 'l2':
            length = 1.0 * math.sqrt(sum(val ** 2 for _, val in vec))
        assert length > 0.0, "sparse documents must not contain any explicit zero entries"
        return ret_normalized_vec(vec, length)
    else:
        raise ValueError("unknown input type")


def argsort(x, topn=None, reverse=False):
    """
    Return indices of the `topn` smallest elements in array `x`, in ascending order.

    If reverse is True, return the greatest elements instead, in descending order.

    """
    x = np.asarray(x)  # unify code path for when `x` is not a np array (list, tuple...)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(np, 'argpartition'):
        return np.argsort(x)[:topn]
    # np >= 1.8 has a fast partial argsort, use that!
    most_extreme = np.argpartition(x, topn)[:topn]
    return most_extreme.take(np.argsort(x.take(most_extreme)))  # resort topn into order


def zeros_aligned(shape, dtype, order='C', align=128):
    """Like `np.zeros()`, but the array will be aligned at `align` byte boundary."""
    nbytes = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
    buffer = np.zeros(nbytes + align, dtype=np.uint8)  # problematic on win64 ("maximum allowed dimension exceeded")
    start_index = -buffer.ctypes.data % align
    return buffer[start_index: start_index + nbytes].view(dtype).reshape(shape, order=order)
