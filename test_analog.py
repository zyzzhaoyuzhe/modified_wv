"""
Perform word analog test on word embedding
"""
import gensim
from gensim.models.keyedvectors import KeyedVectors
import os
from collections import defaultdict
import operator
import fm_ngram


model = gensim.models.Word2Vec.load('model_wv_enwiki-basic_baseline')
print model.wv.most_similar(positive='dog')
# foo1 = model.wv.accuracy('/media/vincent/Data-adhoc/google-analog.txt')
foo2 = model.accuracy('/media/vincent/Data-adhoc/google-analog.txt', most_similar=KeyedVectors.most_similar_cosmul)



# with open('enwiki-basic', 'r') as h:
#     MAXSIZE = 10000000
#     cache = defaultdict(int)
#     thre = 2
#     for idx, line in enumerate(h):
#         if idx % 10000 == 0:
#             print idx
#         for word in line.split(' '):
#             if word.isalpha():
#                 cache[word] += 1
#         firsttimer = 1
#         if len(cache) > MAXSIZE:
#             while len(cache) > 0.5* MAXSIZE:
#                 for word in list(cache):
#                     if cache[word] < thre:
#                         del cache[word]
#                 if not firsttimer:
#                     thre += 1
#                 firsttimer = 0
#     cache = sorted(cache.items(), key=operator.itemgetter(1), reverse=True)[:1000000]
#
#     fout = open('enwiki-vocab', 'w')
#     for word, _ in cache:
#         fout.write(' '.join(word) + '\n')


model = fm_ngram.fm_ngram.load('models/model_PITF-analog_neg10_sgd_enwiki-basic')
model.most_similar(positive='dog')
model.ngram = 1
foo = model.accuracy('/media/vincent/Data-adhoc/google-analog.txt')