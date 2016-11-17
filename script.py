import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import mword2vec
import cPickle as pickle
from helpers import smartfile, get_wordnet_pos
from helpers import inner2prob
from wikicorpus import WikiCorpus
import gensim, nltk
from collections import defaultdict



# text = pickle.load(open('ap.p', 'rb'))
text = smartfile('/media/vincent/Data/Dataset/wiki_en/enwiki-20160920_basic-complete')

# load from file
model = mword2vec.mWord2Vec.load('model_wiki_basic-complete_raw')
model.raw_vocab = defaultdict(int)
for key in model.vocab.iterkeys(): model.raw_vocab[key] = model.vocab[key].count

del model.syn0, model.syn0norm, model.syn1neg

model.scale_vocab()
model.finalize_vocab()
cum_table = model.cum_table

model = mword2vec.mWord2Vec.load('model_wiki_basic-complete_300_1_5_1')
model.cum_table = cum_table
model.clear_sims()
model.init_sims()




# model = mword2vec.mWord2Vec(text, max_vocab_size=1000000, size=300, min_count=1, sample=0,
#                             wPMI=1, smooth_power=1, negative=5, neg_mean=1, weight_power=1,
#                             workers=4,
#                             alpha=0.0025, min_alpha=0.00001, epoch=5, init='gaussian')

# model.build_vocab(text)
model.scale_vocab()
model.finalize_vocab()
model.train(text)

####
model = gensim.models.Word2Vec(text, max_vocab_size=1000000, size=300, min_count=1,
                               sample=0, sg=1, smooth_power=1, negative=1, workers=4, iter=5)
model = gensim.models.Word2Vec(text, max_vocab_size=1000000, size=300, sg=1, workers=4)
#
# model.similar_by_word('soviet')


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
import matplotlib.pyplot as plt

word1 = 'good'
word2 = 'better'
word3 = 'big'
ans = 'bigger'

vec1 = model.wPMI2PMI(model.vocab[word1].index)

dvec1 = model.syn0norm[model.vocab[word1].index] - model.syn0norm[model.vocab[word2].index]
dvec2 = model.syn0norm[model.vocab[word3].index] - model.syn0norm[model.vocab[ans].index]
np.dot(unitvec(dvec1), unitvec(dvec2))
print np.linalg.norm(dvec1), np.linalg.norm(dvec2)


table

vec1 = np.dot(model.syn0[model.vocab[word1].index], model.syn1neg[:50000, :].T)
vec1 =
vec2 = np.dot(model.syn0[model.vocab[word2].index], model.syn1neg[:50000, :].T)
vec3 = np.dot(model.syn0[model.vocab[word3].index], model.syn1neg[:50000, :].T)
vec4 = np.dot(model.syn0[model.vocab[ans].index], model.syn1neg[:50000, :].T)



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
#
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
        foo = model.most_similar(positive=[word3, word2], negative=[word1], topn=30)[0][0]
        htmp.write(' '.join([word1, word2, word3, ans, foo]) + '\n')
        if foo == ans:
            hit+=1
        count+=1
    htmp.close()



#####
model.syn1norm = (model.syn1neg / np.sqrt((model.syn1neg**2).sum(-1))[..., np.newaxis]).astype(float)

words = ['cold/JJS', 'season/NN','ear/NN','polar/JJ','temperate/JJ','climate/NNS','autumn/NN','spring/NN']
vec = unitvec(np.array([model.syn1norm[model.vocab[w].index] for w in words]).mean(axis=0))
# vec = model.syn1norm[model.vocab['altruism']]
print [model.index2word[idx] for idx in np.argsort(np.dot(model.syn0norm, vec[:,np.newaxis]).squeeze())[::-1][:20] if model.index2word[idx] not in words]

