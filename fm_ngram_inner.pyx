#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp, log, sqrt
from libc.string cimport memset
from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport printf, getc
from libc.stdlib cimport exit

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

REAL = np.float32

DEF MAX_SENTENCE_LEN = 10000

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x
cdef dscal_ptr dscal=<dscal_ptr>PyCObject_AsVoidPtr(fblas.dscal._cpointer) # x = alpha * x

cdef void elemul(const int *N, const float *X1, const float *X2, float *Y) nogil:
    """elementwise multiplication: Y = X1 .* X2"""
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i] = X1[i] * X2[i]

cdef void matrix2vec(const int *N, const int *ngram,
                     const np.uint32_t *indices,
                     REAL_t *syn0, REAL_t *inner_cache) nogil:
    """"""
    cdef int i, j
    # initialize inner_cache
    for i in range(ngram[0] * N[0]):
        inner_cache[i] = ONEF
    for i in range(ngram[0]):
        for j in range(ngram[0]):
            if i == j:
                continue
            elemul(N, &inner_cache[j * N[0]], &syn0[indices[i] * ngram[0] * N[0] + i * N[0]], &inner_cache[j * N[0]])

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>sdot(N, X, incX, Y, incY)

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef REAL_t a
    a = <REAL_t>0.0
    for i from 0 <= i < N[0] by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])]


# to support random draws from negative-sampling cum_table
cdef inline unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil:
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

# this quick & dirty RNG apparently matches Java's (non-Secure)Random
# note this function side-effects next_random to set up the next number
cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random

## Function for wPMI implement
# min function for ULL
cdef inline unsigned long long min_ull(unsigned long long a, unsigned long long b) nogil:
    if a > b:
        return b
    else:
        return a

cdef inline REAL_t max_real(REAL_t a, REAL_t b) nogil:
    if a > b:
        return a
    else:
        return b

cdef inline REAL_t min_real(REAL_t a, REAL_t b) nogil:
    if a < b:
        return a
    else:
        return b

cdef unsigned long long word_count(const np.uint32_t word_index, np.uint32_t *cum_table) nogil:
    if word_index == 0:
        return <unsigned long long>cum_table[0]
    else:
        return <unsigned long long>(cum_table[word_index] - cum_table[word_index-1])

###### V2
cdef void get_inner_min(REAL_t alpha, REAL_t beta, REAL_t *inner_min,
                       REAL_t *jcount_min) nogil:
    # Notice: alternative exp(beta-1) + 1
    jcount_min[0] = exp(beta-1)
    inner_min[0] = jcount2inner(alpha, jcount_min[0], beta)

cdef REAL_t jcount_newton(REAL_t inner, REAL_t jcount,
                           REAL_t alpha, REAL_t beta) nogil:
    cdef REAL_t foo = log(jcount) - beta
    return jcount - (jcount * foo - inner / alpha) / (foo + ONEF)

# inner = alpha * x (log(x) - beta)
cdef REAL_t jcount2inner(REAL_t alpha, REAL_t jcount, REAL_t beta) nogil:
    return alpha * jcount * (log(jcount) - beta)

cdef REAL_t inner2jcount(REAL_t inner, REAL_t alpha, REAL_t beta, REAL_t jcount_max, const int niter) nogil:
    cdef REAL_t jcount_min
    cdef REAL_t inner_min
    cdef REAL_t inner_max
    cdef REAL_t inner_C
    cdef REAL_t jcount
    cdef int i
    get_inner_min(alpha, beta, &inner_min, &jcount_min)
    inner_max = jcount2inner(alpha, jcount_max, beta)
    if inner < inner_min:
        return max_real(ONEF, jcount_min)
    elif inner > inner_max:
        return jcount_max
    else:
        #
        jcount = exp(beta)
        if inner > 0:
            inner_C = jcount2inner(alpha, ONEF/alpha, beta)
            if inner > inner_C:
                jcount = max_real(ONEF/alpha, jcount)
        for i in range(niter):
            jcount = jcount_newton(inner, jcount, alpha, beta)
    return jcount if jcount > 1 else ONEF

cdef void rmsprop_update(const int *size, REAL_t *sq_grad, const REAL_t *gamma, const REAL_t *epsilon,
                         REAL_t *syn0, REAL_t *grad, const REAL_t *eta) nogil:
    cdef int k
    for k in range(size[0]):
        sq_grad[k] = gamma[0] * sq_grad[k] + (1-gamma[0]) * grad[k] * grad[k]
        syn0[k] += eta[0] / sqrt(sq_grad[k] + epsilon[0]) * grad[k]

# modified fast_sentence_sg_neg
cdef unsigned long long fast_sentence_neg(
    const int ngram, const int negative,
    const int neg_mean, REAL_t weight_power,
    const long long vocab_size, unsigned long long total_words,
    REAL_t C, np.uint32_t *cum_table,
    REAL_t *syn0, const int size,
    const np.uint32_t *word_indices,
    const int optimizer, REAL_t *sq_grad, REAL_t gamma, REAL_t epsilon,
    const REAL_t eta, REAL_t *sgd_cache, REAL_t *inner_cache,
    unsigned long long next_random, REAL_t *word_locks) nogil:

    cdef long long a
    cdef REAL_t alpha = ONEF / C
    cdef REAL_t beta
    cdef REAL_t logtotal = log(total_words)
    cdef REAL_t jcount_max
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t inner, f, g, label
    cdef int idx
    cdef np.uint32_t *indices = <np.uint32_t*>calloc(ngram, cython.sizeof(np.uint32_t))
    cdef int center_gram

    cdef int sample, tmp, tmp1
    # variables for wPMI
    cdef REAL_t jcount
    cdef REAL_t weight, neg_mean_weight
    cdef unsigned long long domain = 2 ** 31 - 1
    cdef REAL_t logdomain = log(domain)
    cdef REAL_t count_adjust = <REAL_t>total_words/domain
    cdef REAL_t foo

    # reset sgd_cache
    memset(sgd_cache, 0, size * ngram * cython.sizeof(REAL_t))
    # memset(work, 0, size * cython.sizeof(REAL_t))

    for sample in range(ngram * negative+1):
        if sample == 0:
            for tmp in range(ngram):
                indices[tmp] = word_indices[tmp]
            label = ONEF
            neg_mean_weight = ONEF
        else:
            center_gram = sample % ngram
            for tmp in range(ngram):
                indices[tmp] = bisect_left(cum_table, (next_random >> 16) % cum_table[vocab_size-1], 0, vocab_size)
                next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            indices[center_gram] = word_indices[center_gram]
            # if target_index == word_index:
            #     continue
            label = <REAL_t>0.0
            if neg_mean:
                neg_mean_weight = ONEF / <REAL_t>negative / ngram
            else:
                neg_mean_weight = ONEF / ngram
        #
        matrix2vec(&size, &ngram, indices, syn0, inner_cache)
        inner = our_dot(&size, &syn0[indices[0] * ngram * size], &ONE, inner_cache, &ONE)
        beta = logtotal - ngram * logdomain
        jcount_max = <REAL_t>word_count(indices[0], cum_table)
        for tmp in range(ngram):
            beta += log(word_count(indices[tmp], cum_table))
            jcount_max = min_real(jcount_max, <REAL_t>word_count(indices[tmp], cum_table))
        jcount = inner2jcount(inner, alpha, beta, jcount_max, 3)
        weight = alpha * jcount
        # sigmoid(x) = 1 / (1 + exp(-x)) (EXP_TABLE)
        foo = ONEF / weight * inner
        if foo <= -MAX_EXP:
            f = EXP_TABLE[0]
        elif foo >= MAX_EXP:
            f = EXP_TABLE[EXP_TABLE_SIZE-1]
        else:
            idx = <int>((foo + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))
            if idx < 0:
                idx = 0
            elif idx > EXP_TABLE_SIZE-1:
                idx = EXP_TABLE_SIZE-1
            else:
                f = EXP_TABLE[idx]

        # gradient
        if optimizer == 0:
            g = (label - f) * eta / weight *  neg_mean_weight
        elif optimizer == 1:
            g = (label - f) / weight *  neg_mean_weight
        if sample == 0:
            for tmp in range(ngram):
                our_saxpy(&size, &g, &inner_cache[tmp * size], &ONE, &sgd_cache[tmp * size], &ONE)
        else:
            for tmp in range(ngram):
                if tmp == center_gram:
                    our_saxpy(&size, &g, &inner_cache[tmp * size], &ONE, &sgd_cache[tmp * size], &ONE)
                else:
                    if optimizer == 0:
                        our_saxpy(&size, &g, &inner_cache[tmp * size], &ONE, &syn0[indices[tmp] * ngram * size + tmp * size], &ONE)
                    elif optimizer == 1:
                        our_scal(&size, &g, &inner_cache[tmp * size], &ONE)
                        rmsprop_update(&size, &sq_grad[indices[tmp] * ngram * size + tmp * size],
                                       &gamma, &epsilon, &syn0[indices[tmp] * ngram * size + tmp * size],
                                       &inner_cache[tmp * size], &eta)

    for tmp in range(ngram):
        if optimizer == 0:
            our_saxpy(&size, &word_locks[word_indices[tmp]], &sgd_cache[tmp * size], &ONE, &syn0[word_indices[tmp] * ngram * size + tmp * size], &ONE)
        elif optimizer == 1:
            rmsprop_update(&size, &sq_grad[indices[tmp] * ngram * size + tmp * size],
                                       &gamma, &epsilon, &syn0[indices[tmp] * ngram * size + tmp * size],
                                       &sgd_cache[tmp * size], &eta)
    # free memory
    free(indices)
    return next_random

def train_batch(model, sentences, alpha, _sgd_cache, _inner_cache):
    cdef int sample = (model.sample != 0)
    # Use mean for negative sampling or sum
    cdef int ngram = model.ngram
    cdef int negative = model.negative
    cdef int neg_mean = model.neg_mean

    cdef REAL_t weight_power = model.weight_power

    cdef int vocab_size = len(model.vocab)
    cdef unsigned long long total_words = model.words_cumnum
    cdef REAL_t C = model.C
    cdef np.uint32_t *cum_table

    cdef int size = model.layer1_size
    cdef int optimizer = 0
    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *sq_grad = <REAL_t *>(np.PyArray_DATA(model.sq_grad))

    cdef REAL_t _alpha = alpha
    cdef REAL_t _gamma = model.gamma
    cdef REAL_t _epsilon = model.epsilon
    cdef REAL_t *work
    cdef REAL_t *inner_buffer
    cdef REAL_t *sgd_cache
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.syn0_lockf))

    cdef np.uint32_t *ngram_indices = <np.uint32_t*>malloc(ngram * cython.sizeof(np.uint32_t))
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    #
    if model.optimizer == 'rmsprop':
        optimizer = 1
        alpha = model.alpha

    cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
    next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    sgd_cache = <REAL_t *>np.PyArray_DATA(_sgd_cache)
    inner_cache = <REAL_t *>np.PyArray_DATA(_inner_cache)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent or len(sent) < ngram:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            # container
            indexes[effective_words] = word.index
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            if idx_end - idx_start < ngram:
                continue
            for i in range(idx_start, idx_end-ngram+1):
                for j in range(ngram):
                    ngram_indices[j] = indexes[i + j]
                next_random = fast_sentence_neg(
                    ngram, negative,
                    neg_mean, weight_power,
                    vocab_size, total_words, C, cum_table,
                    syn0, size, ngram_indices, optimizer, sq_grad, _gamma, _epsilon,
                    _alpha, sgd_cache, inner_cache, next_random, word_locks)

    free(ngram_indices)
    return effective_words

def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.

    """
    global our_dot
    global our_saxpy
    global our_scal

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        our_dot = our_dot_double
        our_scal = dscal
        our_saxpy = saxpy
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        our_dot = our_dot_float
        our_scal = sscal
        our_saxpy = saxpy
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

FAST_VERSION = init()  # initialize the module
MAX_WORDS_IN_BATCH = MAX_SENTENCE_LEN
