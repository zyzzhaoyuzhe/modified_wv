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



# text = pickle.load(open('ap.p', 'rb'))
text = smartfile('/media/vincent/Data/Dataset/wiki_en/enwiki-20160920_basic-complete')

# load from file
model = mword2vec.mWord2Vec.load('models/model_wiki_basic-complete_1')


model.raw_vocab = defaultdict(int)
for key in model.vocab.iterkeys(): model.raw_vocab[key] = model.vocab[key].count
del model.syn0, model.syn0norm, model.syn1neg
model.scale_vocab()
model.finalize_vocab()
cum_table = model.cum_table

## load benchmark
foo = mword2vec.mWord2Vec.load('models/model_basic-complete_benchmark')
model = mword2vec.mWord2Vec.load('models/model_wiki_basic-complete_vocab')
model.syn0 = copy.deepcopy(foo.syn0)
model.syn1neg = copy.deepcopy(foo.syn1neg)
model.vocab = copy.deepcopy(foo.vocab)
model.index2word = copy.deepcopy(foo.index2word)
model.cum_table = copy.deepcopy(foo.cum_table)
model.clear_sims()
model.init_sims()

### load model for training
model = mword2vec.mWord2Vec.load('models/model_wiki_basic-complete_1')
model.size = 300
model.min_count = 1
model.sample = 0
model.wPMI = 1
model.smooth_power = 1
model.negative = 5
model.neg_mean = 1
model.weight_power = 1
model.workers = 4
model.alpha = 0.0025
model.min_alpha = 0.0001

model.reset_weight()
model.train(text)
