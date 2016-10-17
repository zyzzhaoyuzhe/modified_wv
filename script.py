import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import mword2vec
import cPickle as pickle
from helpers import smartfile
from helpers import inner2prob
from wikicorpus import WikiCorpus
import gensim

# text = pickle.load(open('ap.p', 'rb'))
text = smartfile('/media/vincent/Data/Dataset/wiki_en/enwiki-20160920_basic')


model = mword2vec.mWord2Vec(text, max_vocab_size=1000000, size=100, min_count=1, sample=0,
                            wPMI=1, smooth_power=1, negative=5, neg_mean=0, workers=4,
                            alpha=0.0025, min_alpha=0.00001, epoch=5)
model.build_vocab(text)
model.train(text)

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
model = gensim.models.Word2Vec(text, max_vocab_size=1000000, size=300, min_count=1,
                               sample=0, sg=1, negative=5, workers=4, iter=5)
#
# model.similar_by_word('soviet')
#
#
#
# ## PMI analysis
# import PMI
# import cPickle as pickle
#
# data = pickle.load(open('ap.p','r'))
# vocab_all, freq_all = PMI.getvocab(data)
# vocab, freq = PMI.getvocab(data)
# cofreq = PMI.get_cooccurrance(data, vocab_all, window=5, dynamic_window=True)
#
# pmi = PMI.get_pmi(cofreq)
# wpmi = PMI.get_wpmi(cofreq, pmi, type='wpmi')
#
# # 'wpmi' 'ppwpmi' 'minwpmi'
#
#
#
# data_stop = clear_wordset(data, set(stopwords.words('english')))
# vocab, freq = getvocab(data_stop)
# cofreq = get_cooccurrance(data_stop, vocab, window=5, dynamic_window=True)
#
# data_lowfreq_stop = clear_wordset(data, set(vocab_all[-17000:] + stopwords.words('english')))
# vocab, freq = getvocab(data_lowfreq_stop)
# cofreq = get_cooccurrance(data_lowfreq_stop, vocab, window=5, dynamic_window=True)
#
# data_lowfreq = clear_wordset(data, set(vocab_all[-17000:]))
# vocab, freq = getvocab(data_lowfreq)
# cofreq = get_cooccurrance(data_lowfreq, vocab, window=5, dynamic_window=True)
#
# n_largest(np.tril(wpmi, -1), vocab, 20)
# n_largest(np.tril(pmi2, -1), vocab, 20)
#
# show_context(wpmi, vocab, 'vote', 10)
# show_context(pmi2, vocab, 'prime', 10)