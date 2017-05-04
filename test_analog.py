"""
Perform word analog test on word embedding
"""
import gensim
import os
from collections import defaultdict


model = gensim.models.Word2Vec.load('model_wv_enwiki-basic_baseline')
model.wv.most_similar(positive='dog')
model.wv.accuracy('/media/vincent/Data-adhoc/google-analog.txt')


with open('enwiki-basic', 'r') as h:
    MAXSIZE = 100000
    cache = defaultdict(int)
    thre = 2
    for line in h:
        for word in line.split(' '):
            cache[word] += 1
        firsttimer = 1
        while len(cache) > MAXSIZE:
            for word in list(cache):
                if cache[word] < thre:
                    del cache[word]
            if not firsttimer:
                thre += 1
            firsttimer = 0
    fout = open('enwiki-vocab', 'w')
    for word in cache.iterkeys():
        fout.write(' '.join(word) + '\n')


