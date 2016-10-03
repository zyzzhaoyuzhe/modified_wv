import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import mword2vec
import pickle
from helpers import smartfile
from helpers import inner2prob
from wikicorpus import WikiCorpus

# text = pickle.load(open('ap.p', 'rb'))
text = smartfile('/media/vincent/Data/Dataset/wiki_en/enwiki-20160920')


model = mword2vec.mWord2Vec(text, size=300, min_count=1, sample=0, wPMI=1, smooth_power=1, negative=5, neg_mean=1, workers=4)
# model2 = mword2vec.mWord2Vec(data, min_count=1, sample=0, wPMI=1, smooth_power=1, negative=5, neg_mean=0, workers=1)
#
# model3 = mword2vec.mWord2Vec(data, min_count=1, sample=0, wPMI=0, smooth_power=1, negative=5, neg_mean=1, workers=1)
# model4 = mword2vec.mWord2Vec(data, min_count=1, sample=0, wPMI=0, smooth_power=1, negative=5, neg_mean=0, workers=1)
#
# model5 = mword2vec.mWord2Vec(data, min_count=1, sample=0, wPMI=0, smooth_power=0.75, negative=5, neg_mean=1, workers=1)
#
# for e in model1.similar_by_word('soviet', 10):  print e, model1.vocab[e[0]].count
#
#
#
# model = gensim.models.Word2Vec(data, min_count=1, sample=0, sg=1, negative=5, workers=1)