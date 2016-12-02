import logging, sys
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
import time
logger = logging.getLogger(__name__)

model_file, isnormalize, iswc = sys.argv[1], bool(sys.argv[2]), bool(sys.argv[3])
save_file = '_'.join(['bigram', model_file])
save_file += '_normalized' if isnormalize else '_unnormalized'
save_file += '_wc' if iswc else '_ww'
print model_file, isnormalize, iswc, save_file
# text = pickle.load(open('ap.p', 'rb'))
text = smartfile('/media/vincent/Data/Dataset/wiki_en/enwiki-20160920_basic-complete')

# load from file
model = mword2vec.mWord2Vec.load('models/' + model_file)
model.init_sims()

import heapq
import numpy as np
dic = {}
bigrams = []
topN = 100000
nline = 95638957

for idx, line in enumerate(text):
    if idx % 10000 == 0:
        logger.info('%.2f%% is completed' % (float(idx)/nline * 100))
    for i in range(len(line)-1):
        if line[i] not in model.vocab or line[i+1] not in model.vocab: continue
        if line[i]+ line[i+1] in dic or line[i+1] + line[i] in dic: continue
        if line[i] == line[i+1]: continue
        if iswc:
            sim = model.similarity_wc(line[i], line[i+1], unit=isnormalize) + model.similarity_wc(line[i+1], line[i], unit=isnormalize)
        else:
            sim = model.similarity(line[i], line[i+1], unit=isnormalize) + model.similarity(line[i+1], line[i], unit=isnormalize)
        sim /= 2
        if len(bigrams) < topN:
            heapq.heappush(bigrams, (sim, (line[i], line[i+1])))
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

bigrams = sorted(bigrams, reverse=True)
with open(save_file, 'w') as h:
    for item in bigrams:
        h.write('\t'.join([item[1][0], item[1][1], str(item[0]), str(model.vocab[item[1][0]].count), str(model.vocab[item[1][1]].count)]) + '\n')