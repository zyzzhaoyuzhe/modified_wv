import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import mword2vec
import pickle
from helpers import inner2prob
from wikicorpus import WikiCorpus

# inner2prob(10, [10,10], 100, [0, 0.2], niter=3)
# inner2prob(3522, [3522, 1932, 3, 2, 9, 1031], 2147483647, [-5.70502721e-07, 0.00000000e+00, 0.00000000e+00,
#                                                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00], niter=2)

wiki_file = '/media/vincent/Data/Dataset/wiki_en/enwiki-20160920-pages-articles-multistream.xml'

corpus = WikiCorpus(wiki_file, dictionary=False)

gen = corpus.get_texts()

print gen.next()





# data = pickle.load(open('ap.p','rb'))
#
# model1 = mword2vec.mWord2Vec(data, min_count=1, sample=0, wPMI=1, smooth_power=1, negative=5, neg_mean=1, workers=1)
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