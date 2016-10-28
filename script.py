import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import mword2vec
import cPickle as pickle
from helpers import smartfile, get_wordnet_pos
from helpers import inner2prob
from wikicorpus import WikiCorpus
import gensim, nltk



text = pickle.load(open('ap.p', 'rb'))
# text = smartfile('/media/vincent/Data/Dataset/wiki_en/enwiki-20160920_tag-complete')


model = mword2vec.mWord2Vec(text, max_vocab_size=1000000, size=300, min_count=1, sample=0,
                            wPMI=1, smooth_power=1, negative=5, neg_mean=1, workers=4,
                            alpha=0.0025, min_alpha=0.00001, epoch=5, init='gaussian')
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
model = gensim.models.Word2Vec(text, max_vocab_size=1000000, size=300, sg=1, workers=4)
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

##### Syntatic test
# ans = '/media/vincent/Data/Dataset/Syntactic Test/word_relationship.answers'
# que = "/media/vincent/Data/Dataset/Syntactic Test/word_relationship.questions"
# fsave = '/media/vincent/Data/Dataset/Syntactic Test/word_relationship.pos'
# fa = open(ans, 'r')
# fq = open(que, 'r')
#
# fs = open(fsave,'w')
#
# lmtz = nltk.stem.WordNetLemmatizer()
# for line in fa:
#     tags, foo = tuple(line.split())
#     tag1, tag2 = tuple(tags.split('_'))
#     foo = fq.readline()
#     word1, word2, word3 = tuple(foo.split())
#     word1 = lmtz.lemmatize(word1, get_wordnet_pos(tag1)).encode('utf-8') + '/' + tag1
#     word2 = lmtz.lemmatize(word2, get_wordnet_pos(tag2)).encode('utf-8') + '/' + tag2
#     word4 = lmtz.lemmatize(word3, get_wordnet_pos(tag1)).encode('utf-8') + '/' + tag2
#     word3 = lmtz.lemmatize(word3, get_wordnet_pos(tag1)).encode('utf-8') + '/' + tag1
#
#     fs.write(' '.join([word1, word2, word3, word4]) + '\n')
# fs.close()
# fa.close()
# fq.close()
from gensim.matutils import unitvec
import numpy as np

word1 = 'great/JJ'
word2 = 'great/JJS'
word3 = 'good/JJ'
ans = 'good/JJS'

dvec1 = model.syn0[model.vocab[word1].index] - model.syn0[model.vocab[word2].index]
dvec2 = model.syn0[model.vocab[word3].index] - model.syn0[model.vocab['consider/VBG'].index]
np.dot(unitvec(dvec1), unitvec(dvec2))
print np.linalg.norm(dvec1), np.linalg.norm(dvec2)


model.most_similar(positive=[word3, word2], negative=[word1])
model.most_similar_cosmul(positive=[word3, word2], negative=[word1])
model.most_similar_cosmul_not(positive=[word3, word2], negative=[word1])


####
foo = np.concatenate(([model.cum_table[0]], model.cum_table[1:] - model.cum_table[:-1]))
foo = foo.astype(float)
model.syn0norm = model.syn0 / foo[:, np.newaxis] * foo[foo.size/2]


### similarity for mword2vec
vec1 = model.syn0[model.vocab[word1].index]
vec2 = model.syn0[model.vocab[word2].index]
vec3 = model.syn0[model.vocab[word3].index]
vecans = vec3 + (vec2-vec1) * np.linalg.norm(vec3) / np.linalg.norm(vec1)
foo = np.dot(model.syn0, vecans)
best = np.argsort(foo)[::-1][:13]
result = [(model.index2word[idx], foo[idx]) for idx in best if model.index2word[idx] not in [word1, word2, word3]]
print model.similar_by_vector(unitvec(vecans), restrict_vocab=400000)
print np.dot(unitvec(vecans), model.syn0norm[model.vocab[ans].index])
print np.linalg.norm(vec1), np.linalg.norm(vec3)

ftest = '/media/vincent/Data/Dataset/Syntactic Test/word_relationship.pos'
file_model = 'model/' + 'model_wiki_tag-complete_300_default'
file_model = 'model/' + 'model_wiki_tag-complete_300_1_5_1'

model = mword2vec.mWord2Vec.load(file_model)
count = 0
hit = 0
with open(ftest, 'r') as h:
    htmp = open('tmp', 'w')
    for line in h:
        word1, word2, word3, ans = tuple(line.split())
        if word1 in model.vocab and word2 in model.vocab and word3 in model.vocab and ans in model.vocab:
            pass
        else:
            continue
        ####
        # vec1 = model.syn0norm[model.vocab[word2].index] - model.syn0norm[model.vocab[word1].index]
        # vec2 = model.syn0norm[model.vocab[ans].index] - model.syn0norm[model.vocab[word3].index]
        # score = np.dot(unitvec(vec1), unitvec(vec2))
        # htmp.write(' '.join([word1, word2, word3, ans]) + ' ' + str(score) + '\n')
        #####
        foo, target_tag = nltk.tag.util.str2tuple(word2)
        foo = model.most_similar(positive=[word3, word2], negative=[word1], topn=30)
        pred = None
        for e, score in foo:
            if target_tag in e:
                pred = e
                break
        ## for analysis
        htmp.write(' '.join([word1, word2, word3, ans, pred or 'None']) + '\n')
        count += 1
        if pred == ans:
            hit += 1
    htmp.close()



