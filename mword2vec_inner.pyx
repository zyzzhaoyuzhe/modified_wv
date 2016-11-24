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

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset
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

cdef unsigned long long word_count(const np.uint32_t word_index, np.uint32_t *cum_table, REAL_t count_adjust) nogil:
    if word_index == 0:
        return <unsigned long long>(cum_table[0] * count_adjust)
    else:
        return <unsigned long long>((cum_table[word_index] - cum_table[word_index-1]) * count_adjust)

## old fast_sentence_sg_net
# cdef unsigned long long fast_sentence_sg_neg(
#     const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
#     REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
#     const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
#     unsigned long long next_random, REAL_t *word_locks) nogil:

#     cdef long long a
#     cdef long long row1 = word2_index * size, row2
#     cdef unsigned long long modulo = 281474976710655ULL
#     cdef REAL_t f, g, label
#     cdef np.uint32_t target_index
#     cdef int d
#     memset(work, 0, size * cython.sizeof(REAL_t))
#     for d in range(negative+1):
#         if d == 0:
#             target_index = word_index
#             label = ONEF
#         else:
#             target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
#             next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
#             if target_index == word_index:
#                 continue
#             label = <REAL_t>0.0
#         row2 = target_index * size
#         f = our_dot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
#         if f <= -MAX_EXP or f >= MAX_EXP:
#             continue
#         f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
#         g = (label - f) * alpha
#         our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
#         our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
#     our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)
#     return next_random


###### V2
cdef REAL_t jcount2inner(REAL_t jcounts, unsigned long long count1, unsigned long long count2,
    unsigned long long D, REAL_t C, REAL_t weight_power) nogil:
    return (jcounts / C) ** weight_power * log(<REAL_t>jcounts / count1 * D / count2)


cdef void inner_minmax(unsigned long long count1, unsigned long long count2, 
    unsigned long long D, REAL_t C, REAL_t weight_power,
    REAL_t *jcount_min, REAL_t *inner_min, REAL_t *inner_max) nogil:
    # min and max of inner product, according to the equation
    # #(w,c)/C log \frac{#(w,c)*D}{#(w)#(c)}
    cdef REAL_t foo
    jcount_min[0] = <REAL_t>count1 / D * count2 * exp(-ONEF/weight_power) + 1
    inner_min[0] = jcount2inner(jcount_min[0], count1, count2, D, C, weight_power)
    inner_max[0] = jcount2inner(<REAL_t>min_ull(count1, count2), count1, count2, D, C, weight_power)


cdef REAL_t jcount_newton(
    REAL_t x, unsigned long long count1, unsigned long long count2,
    unsigned long long D, REAL_t C, REAL_t weight_power,
    REAL_t inner) nogil:
    cdef REAL_t foo = log(x / count1 * D / count2)
    return x - (x * foo - inner * C ** weight_power * x ** (ONEF - weight_power)) / (weight_power * foo + ONEF)


cdef REAL_t inner2jcount(
    unsigned long long count1, unsigned long long count2, 
    unsigned long long D, REAL_t C, REAL_t weight_power,
    REAL_t inner, int niter) nogil:
    # Calculate joint count from inner product by the following equation
    # inner = #(w,c)/C log \frac{#(w,c)*D}{#(w)#(c)}
    cdef REAL_t jcount_min, jcount_inde
    cdef REAL_t foo
    cdef REAL_t inner_min, inner_max, inner_C
    inner_minmax(count1, count2, D, C, weight_power, &jcount_min, &inner_min, &inner_max)
    # for debug
    # printf("jcount_min %f, inner_min %f, inner_max %f\n", jcount_min, inner_min, inner_max)
    # printf("inner %f\n", inner)
    if inner < inner_min:
        if jcount_min < 1:
            return ONEF
        else:
            return jcount_min
    elif inner > inner_max:
        return <REAL_t>min_ull(count1, count2)
    else:
        # A smart initialization for Newton
        jcount_inde = <REAL_t>count1 / D * count2
        if inner > 0:
            inner_C = jcount2inner(C, count1, count2, D, C, weight_power)
            if inner > inner_C:
                foo = max_real(C, <REAL_t> jcount_inde)
            else:
                foo = <REAL_t> jcount_inde
        else:
            # starts from gradient transition point; alpha should between (0,0.5)
            foo = (exp(ONEF / (ONE-weight_power) - ONEF/ weight_power ) * jcount_inde)
            if foo > max_real(jcount_inde, C):
                foo = max_real(jcount_inde, C)
        # # for debug
        # printf("inner %f, inner_C %f, jcount_inde %f, C %f \n", inner, inner_C, jcount_inde, C)
        # printf("count1 %d, count2, %d, D %d\n", count1, count2, D)
        # Newton iterations
        for i in range(niter):
            # # for debug
            # printf("foo %f\n", foo)
            
            foo = jcount_newton(foo, count1, count2, D, C, weight_power, inner)
        return foo



# modified fast_sentence_sg_neg
cdef unsigned long long fast_sentence_sg_neg(
    const int negative, const int neg_mean, const int wPMI, REAL_t weight_power,
    const long long vocab_size, unsigned long long total_words, REAL_t C, np.uint32_t *cum_table,
    REAL_t *syn0, REAL_t *syn1neg, const int size, 
    const np.uint32_t word_index, const np.uint32_t word2_index, 
    const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, REAL_t *word_locks) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t inner, f, g, label
    cdef int idx
    cdef np.uint32_t target_index
    cdef int d
    # variables for wPMI mode
    cdef unsigned long long count1, count2
    cdef REAL_t jcounts
    cdef REAL_t weight, neg_mean_weight
    cdef unsigned long long domain = 2 ** 31 - 1
    cdef REAL_t count_adjust = <REAL_t>total_words/domain
    # cdef unsigned long long D = total_words
    # cdef REAL_t C
    cdef REAL_t foo
    
    # for debug
    cdef int i_debug
    cdef REAL_t for_debug = 0
    cdef int ddd = 0

    # if wPMI:
        # C = 344622
        # D = total_words
        # C = (cum_table[<int>(vocab_size*0.5)]-cum_table[<int>(vocab_size*0.5)-1]) * count_adjust
        # printf("C %f, D %d\n",C, D)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
            neg_mean_weight = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[vocab_size-1], 0, vocab_size)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0
            neg_mean_weight = 1.0 / <REAL_t>negative

        row2 = target_index * size
        inner = our_dot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if wPMI:
            count1 = word_count(word2_index, cum_table, count_adjust)
            count2 = word_count(target_index, cum_table, count_adjust)
            # v2
            # jcounts = inner2jcount(count1, count2, D, C, inner, 3)
            # weight = jcounts / C
            # v3
            jcounts = inner2jcount(count1, count2, 
                total_words, C, weight_power, 
                inner, 3)
            weight = (jcounts / C) ** weight_power


            # for debug
            # printf("c1 %d, c2 %d, inner %f, jc %f, C %f, D %d, weight %f\n", count1, count2, inner, jcounts, C, total_words, weight)
            # exit(0)
            # weight = 0.1

            foo = ONEF / weight * inner
            # foo = inner
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
                # f = EXP_TABLE[<int>((inner + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            g = (label - f) * alpha / weight
            # g = (label - f) * alpha
            if neg_mean:
                g = g * neg_mean_weight
        else:
            if inner <= -MAX_EXP or inner >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((inner + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)

        # # debug
        # for_debug = 0
        # for i_debug in range(size):
        #     for_debug += syn1neg[row2+i_debug] * syn1neg[row2+i_debug]
        # if g>0:
        #     printf("%d, %d, jcount %d, D %d, C %f, weight %f\n", count1, count2, jcounts, D, C, weight)
        #     printf("g: %f, inner %f\n", g, inner)
        #     printf("norm: %f\n", for_debug)
        #     for_debug = 0
        #     for i_debug in range(size):
        #         for_debug += syn0[row1+i_debug] * syn0[row1+i_debug]
        #     printf("work norm: %f\n", for_debug)
        #     # exit(0)

    our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)

    return next_random

###### V1

# cdef REAL_t jcount2inner(unsigned long long jcounts, unsigned long long count1, unsigned long long count2,
#     unsigned long long D, REAL_t C) nogil:
#     return <REAL_t>jcounts / C * log(<REAL_t>jcounts / count1 * D / count2)


# cdef void inner_minmax(unsigned long long count1, unsigned long long count2, unsigned long long D, REAL_t C,
#     unsigned long long *jcount_min, REAL_t *inner_min, REAL_t *inner_max) nogil:
#     # min and max of inner product, according to the equation
#     # #(w,c)/C log \frac{#(w,c)*D}{#(w)#(c)}
#     cdef REAL_t foo
#     jcount_min[0] = <unsigned long long>(count1 / <REAL_t> D * count2 * exp(-ONEF) + 1)
#     inner_min[0] = jcount2inner(jcount_min[0], count1, count2, D, C)
#     inner_max[0] = jcount2inner(min_ull(count1, count2), count1, count2, D, C)


# cdef REAL_t jcount_newton(REAL_t x, unsigned long long count1, unsigned long long count2,
#     unsigned long long D, REAL_t C, REAL_t inner) nogil:
#     cdef REAL_t foo = log(<REAL_t>x / count1 * D / count2)
#     return x - (x * foo - inner * C) / (foo + ONEF)


# cdef unsigned long long inner2jcount(unsigned long long count1, unsigned long long count2, 
#     unsigned long long D, REAL_t C, REAL_t inner, int niter) nogil:
#     # Calculate joint count from inner product by the following equation
#     # inner = #(w,c)/C log \frac{#(w,c)*D}{#(w)#(c)}
#     cdef unsigned long long jcount_min
#     cdef REAL_t foo
#     cdef REAL_t inner_min
#     cdef REAL_t inner_max
#     inner_minmax(count1, count2, D, C, &jcount_min, &inner_min, &inner_max)
#     if inner < inner_min:
#         if jcount_min < 1:
#             return 1
#         else:
#             return jcount_min
#     elif inner > inner_max:
#         return min_ull(count1, count2)
#     else:
#         foo = min_ull(count1, count2)
#         for i in range(niter):
#             foo = jcount_newton(foo, count1, count2, D, C, inner)
#         if foo < 1:
#             return 1
#         else:
#             return <unsigned long long>foo

# ## modified fast_sentence_sg_net
# cdef unsigned long long fast_sentence_sg_neg(
#     const int negative, const int neg_mean, const int wPMI, REAL_t weight_power,
#     const long long vocab_size, unsigned long long total_words, np.uint32_t *cum_table,
#     REAL_t *syn0, REAL_t *syn1neg, const int size, 
#     const np.uint32_t word_index, const np.uint32_t word2_index, 
#     const REAL_t alpha, REAL_t *work,
#     unsigned long long next_random, REAL_t *word_locks) nogil:

#     cdef long long a
#     cdef long long row1 = word2_index * size, row2
#     cdef unsigned long long modulo = 281474976710655ULL
#     cdef REAL_t inner, f, g, label
#     cdef int idx
#     cdef np.uint32_t target_index
#     cdef int d
#     # variables for wPMI mode
#     cdef unsigned long long count1, count2, jcounts
#     cdef REAL_t weight, neg_mean_weight
#     cdef unsigned long long domain = 2 ** 31 - 1
#     cdef REAL_t count_adjust = <REAL_t>total_words/domain
#     cdef unsigned long long D
#     cdef REAL_t C
#     cdef REAL_t foo
#     cdef int i_debug
#     cdef REAL_t for_debug = 0

#     if wPMI:
#         # C = 344622
#         D = total_words
#         C = (cum_table[<int>(vocab_size*0.8)]-cum_table[<int>(vocab_size*0.8)-1]) * count_adjust
#         # printf("C %f\n",C)

#     memset(work, 0, size * cython.sizeof(REAL_t))

#     for d in range(negative+1):
#         if d == 0:
#             target_index = word_index
#             label = ONEF
#             neg_mean_weight = ONEF
#         else:
#             target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[vocab_size-1], 0, vocab_size)
#             next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
#             if target_index == word_index:
#                 continue
#             label = <REAL_t>0.0
#             neg_mean_weight = 1.0 / <REAL_t>negative

#         row2 = target_index * size
#         inner = our_dot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
#         if wPMI:
#             count1 = word_count(word2_index, cum_table, count_adjust)
#             count2 = word_count(target_index, cum_table, count_adjust)
#             # v2
#             jcounts = inner2jcount(count1, count2, D, C, inner, 3)
#             weight = jcounts / C
#             # v3
#             # jcounts = inner2jcount(count1, count2, D, C / D * count1, inner, 3)
#             # weight = jcounts / C * D / count1


#             # for debug
#             # printf("c1 %d, c2 %d, inner %f, jc %d, C %f, D %d, weight %f\n", count1, count2, inner, jcounts, C, D, weight)
#             # exit(0)
#             # weight = 0.1

#             foo = ONEF / weight * inner
#             # foo = inner
#             if foo <= -MAX_EXP:
#                 f = EXP_TABLE[0]
#             elif foo >= MAX_EXP:
#                 f = EXP_TABLE[EXP_TABLE_SIZE-1]
#             else:
#                 idx = <int>((foo + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))
#                 if idx < 0:
#                     idx = 0
#                 elif idx > EXP_TABLE_SIZE-1:
#                     idx = EXP_TABLE_SIZE-1
#                 else:
#                     f = EXP_TABLE[idx]
#                 # f = EXP_TABLE[<int>((inner + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
#             g = (label - f) * alpha / weight
#             # g = (label - f) * alpha
#             if neg_mean:
#                 g = g * neg_mean_weight
#         else:
#             if inner <= -MAX_EXP or inner >= MAX_EXP:
#                 continue
#             f = EXP_TABLE[<int>((inner + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
#             g = (label - f) * alpha
#         our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
#         our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)

#         # # debug
#         # for_debug = 0
#         # for i_debug in range(size):
#         #     for_debug += syn1neg[row2+i_debug] * syn1neg[row2+i_debug]
#         # if g>0:
#         #     printf("%d, %d, jcount %d, D %d, C %f, weight %f\n", count1, count2, jcounts, D, C, weight)
#         #     printf("g: %f, inner %f\n", g, inner)
#         #     printf("norm: %f\n", for_debug)
#         #     for_debug = 0
#         #     for i_debug in range(size):
#         #         for_debug += syn0[row1+i_debug] * syn0[row1+i_debug]
#         #     printf("work norm: %f\n", for_debug)
#         #     # exit(0)

#     our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)

#     return next_random



def train_batch_sg(model, sentences, alpha, _work):
    # cdef int hs = model.hs
    cdef int sample = (model.sample != 0)
    # Use mean for negative sampling or sum
    cdef int negative = model.negative
    cdef int neg_mean = model.neg_mean
    # Use wPMI or PMI
    cdef int wPMI = model.wPMI
    cdef REAL_t weight_power = model.weight_power

    cdef int vocab_size = len(model.vocab)
    cdef unsigned long long total_words = model.words_cumnum
    cdef REAL_t C = model.C
    cdef np.uint32_t *cum_table

    cdef int size = model.layer1_size
    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1neg

    cdef REAL_t _alpha = alpha
    cdef REAL_t *work
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.syn0_lockf))

    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # for debug
    cdef int ddd = 0

    # # For hierarchical softmax
    # cdef int codelens[MAX_SENTENCE_LEN]
    # cdef REAL_t *syn1
    # cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    # cdef np.uint8_t *codes[MAX_SENTENCE_LEN]
    # if hs:
    #     syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            indexes[effective_words] = word.index
            # if hs:
            #     codelens[effective_words] = <int>len(word.code)
            #     codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
            #     points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
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

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - window + reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + window + 1 - reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                for j in range(j, k):
                    if j == i:
                        continue
                    # if hs:
                    #     fast_sentence_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], _alpha, work, word_locks)
                    if negative:
                        ## olde
                        # next_random = fast_sentence_sg_neg(negative, cum_table, cum_table_len, syn0, syn1neg, size, indexes[i], indexes[j], _alpha, work, next_random, word_locks)
                        ## modified 
                        next_random = fast_sentence_sg_neg(
                            negative, neg_mean, wPMI, weight_power,
                            vocab_size, total_words, C, cum_table, 
                            syn0, syn1neg, 
                            size, indexes[i], indexes[j], 
                            _alpha, work, next_random, word_locks)
                        # # for debug
                        # ddd += 1
                        # if ddd > 2:
                        #     exit(0)

    return effective_words


# Score is only implemented for hierarchical softmax
# def score_sentence_sg(model, sentence, _work):

#     cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
#     cdef REAL_t *work
#     cdef int size = model.layer1_size

#     cdef int codelens[MAX_SENTENCE_LEN]
#     cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
#     cdef int sentence_len
#     cdef int window = model.window

#     cdef int i, j, k
#     cdef long result = 0

#     cdef REAL_t *syn1
#     cdef np.uint32_t *points[MAX_SENTENCE_LEN]
#     cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

#     syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

#     # convert Python structures to primitive types, so we can release the GIL
#     work = <REAL_t *>np.PyArray_DATA(_work)

#     vlookup = model.vocab
#     i = 0
#     for token in sentence:
#         word = vlookup[token] if token in vlookup else None
#         if word is None:
#             continue  # should drop the
#         indexes[i] = word.index
#         codelens[i] = <int>len(word.code)
#         codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
#         points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
#         result += 1
#         i += 1
#         if i == MAX_SENTENCE_LEN:
#             break  # TODO: log warning, tally overflow?
#     sentence_len = i

#     # release GIL & train on the sentence
#     work[0] = 0.0

#     with nogil:
#         for i in range(sentence_len):
#             if codelens[i] == 0:
#                 continue
#             j = i - window
#             if j < 0:
#                 j = 0
#             k = i + window + 1
#             if k > sentence_len:
#                 k = sentence_len
#             for j in range(j, k):
#                 if j == i or codelens[j] == 0:
#                     continue
#                 score_pair_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], work)

#     return work[0]


# cdef void score_pair_sg_hs(
#     const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
#     REAL_t *syn0, REAL_t *syn1, const int size,
#     const np.uint32_t word2_index, REAL_t *work) nogil:

#     cdef long long b
#     cdef long long row1 = word2_index * size, row2, sgn
#     cdef REAL_t f

#     for b in range(codelen):
#         row2 = word_point[b] * size
#         f = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
#         sgn = (-1)**word_code[b] # ch function: 0-> 1, 1 -> -1
#         f = sgn*f
#         if f <= -MAX_EXP or f >= MAX_EXP:
#             continue
#         f = LOG_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
#         work[0] += f


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.

    """
    global our_dot
    global our_saxpy

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
        our_saxpy = saxpy
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        our_dot = our_dot_float
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
