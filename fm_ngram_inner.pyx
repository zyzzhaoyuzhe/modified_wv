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
cimport scipy.linalg.cython_blas as blas

from libc.math cimport exp, log, sqrt, fabs
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

# cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef scopy_ptr scopy=<scopy_ptr>blas.scopy  # y = x
# cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef saxpy_ptr saxpy=<saxpy_ptr>blas.saxpy  # y += alpha * x
# cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef sdot_ptr sdot=<sdot_ptr>blas.sdot  # float = dot(x, y)
# cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>blas.dsdot  # double = dot(x, y)
# cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef snrm2_ptr snrm2=<snrm2_ptr>blas.snrm2
# cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x
cdef sscal_ptr sscal=<sscal_ptr>blas.sscal # x = alpha * x
# cdef dscal_ptr dscal=<dscal_ptr>PyCObject_AsVoidPtr(fblas.dscal._cpointer) # x = alpha * x
cdef dscal_ptr dscal=<dscal_ptr>blas.dscal # x = alpha * x
cdef ssbmv_ptr ssbmv=<ssbmv_ptr>blas.ssbmv # y = alpha * AX + beta * Y

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef int ZERO = 0
cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t ZEROF = <REAL_t>0.0
cdef char UPLO = 'L'

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

# cdef void elemul(const int *N, const float *X1, const float *X2, float *Y) nogil:
#     """elementwise multiplication: Y = X1 .* X2"""
#     cdef int i
#     for i in range(N[0]):
#         Y[i] = X1[i] * X2[i]

cdef void matrix2vec(const int *N, const int *ngram,
                     const np.uint32_t *indices,
                     const REAL_t *syn0, REAL_t *inner_cache) nogil:
    """"""
    cdef int i, j, k
    # cdef REAL_t foo[1000]
    # # initialize inner_cache
    # for i in range(ngram[0] * N[0]):
    #     inner_cache[i] = ONEF
    # no_blas
    for i in range(ngram[0]):
        for k in range(N[0]):
            inner_cache[i*N[0]+k] = ONEF
            for j in range(ngram[0]):
                if j == i:
                    continue
                # inner_cache[i*N[0]+k] *= syn0[j * N[0] + k]
                inner_cache[i*N[0]+k] *= syn0[indices[j] * N[0] * ngram[0] + j * N[0] + k]
    # # blas
    # for i in range(ngram[0]):
    #     for j in range(ngram[0]):
    #         if j == i:
    #             continue
    #         # printf('i: %d - j: %d\n', i, j)
    #         # printf('%f - %f\n', syn0[indices[i] * N[0] * ngram[0] + i * N[0]+1], inner_cache[j*N[0]+1])
    #         our_sbmv(&UPLO, N, &ZERO, &ONEF, &syn0[indices[i] * N[0] * ngram[0] + i * N[0]], &ONE,
    #                  &inner_cache[j*N[0]], &ONE, &ZEROF, foo, &ONE)
    #         scopy(N, foo, &ONE, &inner_cache[j*N[0]], &ONE)
    #         # printf('%f\n', inner_cache[j*N[0]+1])

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
    # return jcount if jcount > 1 else ONEF
    return jcount

cdef void rmsprop_update(const int *size, REAL_t *sq_grad, const REAL_t *gamma,
                         const REAL_t *epsilon, const REAL_t *eta,
                         REAL_t *grad, REAL_t *syn0) nogil:
    cdef int k
    cdef REAL_t inner, weight
    inner = our_dot(size, grad, &ONE, grad, &ONE)
    sq_grad[0] = gamma[0] * sq_grad[0] + (1-gamma[0]) * inner
    weight = eta[0] / sqrt(sq_grad[0] + epsilon[0])
    our_saxpy(size, &weight, grad, &ONE, syn0, &ONE)

# modified fast_sentence_sg_neg
cdef unsigned long long fast_sentence_neg(
    const int ngram, const int negative,
    const int neg_mean, REAL_t weight_power,
    const long long vocab_size, unsigned long long total_words,
    REAL_t C, np.uint32_t *cum_table,
    REAL_t *syn0, const int size,
    const np.uint32_t *word_indices,
    const int optimizer, REAL_t *sq_grad, REAL_t gamma, REAL_t epsilon, const REAL_t eta,
    REAL_t *sgd_cache, REAL_t *inner_cache, REAL_t *syn0_copy,
    unsigned long long next_random, REAL_t *word_locks) nogil:

    # Constants
    cdef REAL_t alpha = ONEF / C
    cdef REAL_t logtotal = log(total_words)
    cdef unsigned long long domain = 2 ** 31 - 1
    cdef REAL_t logdomain = log(domain)
    cdef REAL_t beta
    cdef int sizexngram = size * ngram

    # variable for calculate gradients
    cdef int sample, tmp, tmp1
    cdef REAL_t jcount_max
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t inner, f, g, label
    cdef int idx
    cdef int center_gram
    cdef REAL_t jcount
    cdef REAL_t weight, neg_mean_weight

    # variables for wPMI
    cdef REAL_t count_adjust = <REAL_t>total_words/domain
    cdef REAL_t foo
    # index caches
    cdef np.uint32_t *indices = <np.uint32_t*>calloc(ngram, cython.sizeof(np.uint32_t))
    cdef np.uint32_t *neg_indices = <np.uint32_t*>calloc(ngram, cython.sizeof(np.uint32_t))

    # reset sgd_cache
    memset(sgd_cache, 0, size * ngram * cython.sizeof(REAL_t))
    # memset(work, 0, size * cython.sizeof(REAL_t))
    for sample in range(2 * negative+1):
        if sample == 0:
            for tmp in range(ngram):
                indices[tmp] = word_indices[tmp]
            label = ONEF
            neg_mean_weight = ONEF
        else:
            center_gram = sample % ngram
            # center_gram = 0
            for tmp in range(ngram):
                if tmp == center_gram:
                    indices[tmp] = word_indices[tmp]
                else:
                    while True:
                        indices[tmp] = bisect_left(cum_table, (next_random >> 16) % cum_table[vocab_size-1], 0, vocab_size)
                        next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
                        if indices[tmp] != word_indices[tmp]:
                            break

            # indices[1] = bisect_left(cum_table, (next_random >> 16) % cum_table[vocab_size-1], 0, vocab_size)
            # next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            # indices[0] = word_indices[0]

            label = <REAL_t>0.0
            # if neg_mean:
            #     neg_mean_weight = ONEF / <REAL_t>negative / ngram
            # else:
            #     neg_mean_weight = ONEF / ngram

            neg_mean_weight = ONEF
        # syn0 copy to prevent syn0 changed by other thread (there is no lock.)
        # for tmp in range(ngram):
        #     for tmp1 in range(size):
        #         syn0_copy[tmp * size + tmp1] = syn0[indices[tmp]*size*ngram+tmp*size+tmp1]
        matrix2vec(&size, &ngram, indices, syn0, inner_cache)
        inner = our_dot(&size, &syn0[indices[0] * ngram * size], &ONE, &inner_cache[0], &ONE)

        # inner = our_dot(&size, &syn0_copy[0], &ONE, &syn0_copy[size], &ONE)

        # jcount_max = <REAL_t>word_count(indices[0], cum_table)
        # beta = logtotal - ngram * logdomain
        # for tmp in range(ngram):
        #     beta += log(word_count(indices[tmp], cum_table))
        #     jcount_max = min_real(jcount_max, <REAL_t>word_count(indices[tmp], cum_table))
        # jcount = inner2jcount(inner, alpha, beta, jcount_max, 3)
        # weight = alpha * jcount

        weight = ONEF

        # sigmoid(x) = 1 / (1 + exp(-x)) (EXP_TABLE)
        foo = ONEF / weight * inner

        # critical
        if foo <= -MAX_EXP or foo >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((foo + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        # # calculate f
        # if foo <= -MAX_EXP:
        #     f = EXP_TABLE[0]
        # elif foo >= MAX_EXP:
        #     f = EXP_TABLE[EXP_TABLE_SIZE-1]
        # else:
        #     f = EXP_TABLE[<int>((foo + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        # gradient
        if optimizer == 0:
            g = (label - f) * eta / weight *  neg_mean_weight
        elif optimizer == 1:
            g = (label - f) / weight *  neg_mean_weight

        # g = (label - f) * eta
        # our_saxpy(&size, &g, &syn0_copy[size], &ONE, &sgd_cache[0], &ONE)
        # our_saxpy(&size, &g, &syn0_copy[0], &ONE, &syn0[indices[1] * ngram * size + size], &ONE)

        if sample == 0:
            for tmp in range(ngram):
                our_saxpy(&size, &g, &inner_cache[tmp * size], &ONE, &sgd_cache[tmp * size], &ONE)
            # our_saxpy(&size, &g, &inner_cache[0], &ONE, &sgd_cache[0], &ONE)
            # our_saxpy(&size, &g, &inner_cache[size], &ONE, &sgd_cache[size], &ONE)
        else:
            for tmp in range(ngram):
                if tmp == center_gram:
                    our_saxpy(&size, &g, &inner_cache[tmp * size], &ONE, &sgd_cache[tmp * size], &ONE)
                else:
                    if optimizer == 0:
                        our_saxpy(&size, &g, &inner_cache[tmp * size], &ONE, &syn0[indices[tmp] * ngram * size + tmp * size], &ONE)
                    elif optimizer == 1:
                        rmsprop_update(&size, &sq_grad[indices[tmp] * ngram + tmp],
                                       &gamma, &epsilon, &eta,
                                       &inner_cache[tmp * size],
                                       &syn0[indices[tmp] * ngram * size + tmp * size])
            # our_saxpy(&size, &g, &inner_cache[0], &ONE, &sgd_cache[0], &ONE)
            # our_saxpy(&size, &g, &inner_cache[size], &ONE, &syn0[indices[1] * ngram * size + size], &ONE)

    for tmp in range(ngram):
        if optimizer == 0:
            our_saxpy(&size, &word_locks[word_indices[tmp]], &sgd_cache[tmp * size], &ONE, &syn0[word_indices[tmp] * ngram * size + tmp * size], &ONE)
        elif optimizer == 1:
            rmsprop_update(&size, &sq_grad[word_indices[tmp] * ngram + tmp],
                           &gamma, &epsilon, &eta,
                           &sgd_cache[tmp * size],
                           &syn0[word_indices[tmp] * ngram * size + tmp * size])

    # free memory
    free(indices)
    free(neg_indices)
    return next_random

def train_batch(model, sentences, alpha, _sgd_cache, _inner_cache, _syn0_copy):
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
    cdef REAL_t *sgd_cache, *inner_cache, *syn0_copy
    cdef np.uint32_t *indices, *neg_indices
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.syn0_lockf))

    cdef np.uint32_t ngram_indices[10]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    #
    if model.optimizer == 'rmsprop':
        optimizer = 1
        # alpha = model.alpha
    elif model.optimizer == 'sgd':
        optimizer = 0

    cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
    next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)
    # next_random = 1816045175

    # convert Python structures to primitive types, so we can release the GIL
    sgd_cache = <REAL_t *>np.PyArray_DATA(_sgd_cache)
    inner_cache = <REAL_t *>np.PyArray_DATA(_inner_cache)
    syn0_copy = <REAL_t *>np.PyArray_DATA(_syn0_copy)


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
            # if sample and word.sample_int < random_int32(&next_random):
            #     continue
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
            # for i in range(idx_start, idx_end):
                for j in range(ngram):
                    ngram_indices[j] = indexes[i + j]
                next_random = fast_sentence_neg(
                    ngram, negative,
                    neg_mean, weight_power,
                    vocab_size, total_words, C, cum_table,
                    syn0, size, ngram_indices, optimizer, sq_grad, _gamma, _epsilon,
                    _alpha, sgd_cache, inner_cache, syn0_copy,
                    next_random, word_locks)
    return effective_words

def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.

    """
    global our_dot
    global our_saxpy
    global our_scal
    global our_sbmv

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
        our_sbmv = ssbmv
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        our_dot = our_dot_float
        our_scal = sscal
        our_saxpy = saxpy
        our_sbmv = ssbmv
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

FAST_VERSION = init()  # initialize the module
MAX_WORDS_IN_BATCH = MAX_SENTENCE_LEN
