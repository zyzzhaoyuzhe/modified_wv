import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import mword2vec
import cPickle as pickle
from helpers import smartfile, get_wordnet_pos
from helpers import inner2prob
from wikicorpus import WikiCorpus
import gensim, nltk, copy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


# text = pickle.load(open('ap.p', 'rb'))
text = smartfile('../DATA/enwiki-20160920_basic-complete')

# load from file
model = mword2vec.mWord2Vec.load('models/model_wiki_basic-complete_2')
model.init_sims()

import heapq
import numpy as np
dic = {}
bigrams = []
topN = 100000
fin = smartfile('../DATA/enwiki-20160920_basic-complete')
nline = 0

print 'Count number of lines in the corpus!'
for line in fin:
    nline += 1

for idx, line in enumerate(fin):
    if idx % 10000 == 0:
        logger.info('%.2f%% is completed' % (float(idx)/nline * 100))
    for i in range(len(line)-1):
        if line[i] not in model.vocab or line[i+1] not in model.vocab: continue
        if line[i]+ line[i+1] in dic or line[i+1] + line[i] in dic: continue
        vec1a = model.syn0norm[model.vocab[line[i]].index]
        vec2a = model.syn1norm[model.vocab[line[i+1]].index]
        vec1b = model.syn0norm[model.vocab[line[i]].index]
        vec2b = model.syn1norm[model.vocab[line[i + 1]].index]
        sim = (np.dot(vec1a, vec2a) + np.dot(vec1b, vec2b)) / 2.
        if len(bigrams) < topN:
            heapq.heappush(bigrams, (sim,(line[i], line[i+1])))
            dic[line[i] + line[i+1]] = 0
        else:
            foo = heapq.heappop(bigrams)
            dic.pop(foo[1][0] + foo[1][1], None)
            dic.pop(foo[1][1] + foo[1][0], None)
            if foo[0] < sim:
                heapq.heappush(bigrams, (sim,(line[i], line[i+1])))
                dic[line[i] + line[i+1]] = 0
            else:
                heapq.heappush(bigrams, foo)
                dic[''.join(foo[1])] = 0

with open('bigram.txt', 'w') as h:
    for item in bigrams:
        h.write('\t'.join([item[1][0], item[1][1], str(item[0]), model.vocab[item[1][0]].count, model.vocab[item[1][1]].count]))
