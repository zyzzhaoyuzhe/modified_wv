from BeautifulSoup import BeautifulSoup as bs
import re, sys
import nltk
from nltk.corpus import wordnet, stopwords
import numpy as np
import cPickle as pickle


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'


def getvocab(data):
    freq = {}
    vocab = set()
    for senc in data:
        for token in senc:
            if token not in freq:
                freq[token] = 1
                vocab.add(token)
            else:
                freq[token] += 1
    vocab = sorted(vocab, key=freq.__getitem__, reverse=True)
    return vocab, freq


def pre_process(file):
    lmtz = nltk.stem.WordNetLemmatizer()
    txt = open(file, 'r').read()
    soup = bs(txt)
    all = soup.findAll('doc')
    data = []
    for element in all:
        text = element.find('text').text.encode('utf8').lower()
        text = text.replace('-', ' ')
        sents = nltk.sent_tokenize(text)
        for sent in sents:
            tokens = nltk.word_tokenize(sent)
            tags = nltk.pos_tag(tokens)
            foo = []
            for tag in tags:
                token = tag[0]
                treebank_tag = tag[1]
                token = lmtz.lemmatize(token, get_wordnet_pos(treebank_tag)).encode('utf8')
                if re.findall(r'[a-z]', token) and token != "'s":
                    foo.append(token)
            data.append(foo)
    return data


def get_cooccurrance(data, vocab, window=5, dynamic_window=False):
    nwords = len(vocab)
    output = np.zeros([nwords, nwords])
    count = 0.0
    nsents = len(data)
    for sent in data:
        # sys.stdout.write("Progress {0}%\r".format(count/nsents*100))
        # sys.stdout.flush()
        count += 1
        for i, word in enumerate(sent):
            left = max(0, i - window)
            right = min(len(sent) - 1, i + window)
            idxa = vocab.index(word)
            for j in range(left, right + 1):
                if i != j:
                    idxb = vocab.index(sent[j])
                    if dynamic_window:
                        # output[idxa, idxb] += 0.1
                        output[idxa, idxb] += (window - abs(j - i) + 1) / float(window)
                        # pass
                    else:
                        output[idxa, idxb] += 1
                        # pass
    return output


# weighted PMI
def get_wpmi(cofreq, pmi, type='wpmi',power=1):
    D = cofreq.sum()
    word_prob = cofreq.sum(axis=0) / float(D)
    if type == 'wpmi':
        return np.power(cofreq / D, power) * pmi
    elif type == 'ppwpmi':
        return word_prob * word_prob[:, np.newaxis] * pmi
    elif type == 'minwpmi':
        return np.minimum(word_prob, word_prob[:, np.newaxis]) * pmi
    else:
        pass


def get_pmi(cofreq):
    D = cofreq.sum()
    word_freq = cofreq.sum(axis=0)
    m = np.log(cofreq / word_freq / word_freq[:, np.newaxis]) + np.log(D)
    m[np.invert(np.isfinite(m))] = -1000
    return m


def get_pmi2(cofreq):
    D = cofreq.sum()
    word_freq = cofreq.sum(axis=0)
    m = np.log(np.power(cofreq, 2) / word_freq / word_freq[:, np.newaxis]) + np.log(D)
    m[np.invert(np.isfinite(m))] = -1000
    return m


# return top N largest word-context couple
def n_largest(matrix, vocab, n):
    foo = np.argsort(matrix.flatten())[-n:][::-1]
    idxs = np.unravel_index(foo, matrix.shape)
    for k in range(len(idxs[0])):
        print (vocab[idxs[0][k]], vocab[idxs[1][k]], matrix[idxs[0][k], idxs[1][k]])


def clear_wordset(data, words_to_clear):
    output = []
    # words_to_clear = stopwords.words('english')
    for sent in data:
        foo = []
        for word in sent:
            if word not in words_to_clear:
                foo.append(word)
        output.append(foo)
    return output


def show_context(matrix, vocab, freq, word, topn=10):
    """
    Show top n associated contexts of the given word.
    :param word:
    :param topn:
    :return:
    """
    word_idx = vocab.index(word)
    vec = matrix[word_idx, :]
    idxs = np.argsort(vec)[-topn:][::-1]
    for idx in idxs:
        print (word, vocab[idx], freq[vocab[idx]], matrix[word_idx, idx])

def wPMI2PMI(inner, C, D, freq, widx,iter=3):
    output = np.log(D/np.maximum(freq[widx], freq))
    tmp = inner * C / freq[widx] * D / freq
    for _ in xrange(iter):
        output -= (output * np.log(output) - tmp)/(1 + np.log(output))
    return np.log(output)


vocab, idx2vocab = readvocab('/Users/vzhao/Documents/Weighted MI/DATA/ap/vocab.txt')
file = 'ap.txt'
data = pre_process(file)
# data = pickle.load(open('ap.p','r'))
vocab_all, freq_all = getvocab(data)
vocab, freq = getvocab(data)

freq = np.array([freq[word] for word in vocab])

cofreq = get_cooccurrance(data, vocab_all, window=5, dynamic_window=True)
freq = cofreq.sum(axis=1)

pmi = get_pmi(cofreq)
del pmi
pmi = get_wpmi(cofreq, pmi_old, type='wpmi', power=0.1)

import numpy as np
import matplotlib.pyplot as plt
from gensim.matutils import unitvec
import copy

word1 = 'new'
word2 = 'york'
pmi[pmi<0] = 0

vecnorm = np.sqrt(np.sum(pmi*pmi, axis=1))

foo = np.argsort(np.dot((pmi[vocab.index(word1)]/vecnorm[vocab.index(word1)])[np.newaxis, :], pmi/vecnorm[np.newaxis, :]))

print [vocab[idx] for idx in np.argsort(np.dot((pmi[vocab.index(word1)]/vecnorm[vocab.index(word1)])[np.newaxis, :], pmi/(0.0001+vecnorm[np.newaxis, :])).squeeze())[::-1]]
print [vocab[idx] for idx in np.argsort(pmi[vocab.index(word1), :])[::-1]]

plt.figure()
plt.subplot(2,1,1)
vec1 = pmi[vocab.index(word1), :]
vec2 = pmi[vocab.index(word2), :]
print np.dot(unitvec(vec1), unitvec(vec2))
plt.plot(unitvec(vec1[:500]))
plt.plot(unitvec(vec2[:500]))
plt.subplot(2,1,2)
plt.plot(cofreq[vocab.index(word1), :500]/freq[vocab.index(word1)])
plt.plot(cofreq[vocab.index(word2), :500]/freq[vocab.index(word2)])

##
vec1 = pmi[vocab.index(word1), :] * cofreq[vocab.index(word1), :] / cofreq.sum()
idx = vec1 < 1e-7
vec1 = copy.copy(pmi[vocab.index(word1), :])
vec1[idx] = 0
vec2 = pmi[vocab.index(word2), :] * cofreq[vocab.index(word2), :] / cofreq.sum()
idx = vec2 < 1e-7
vec2 = copy.copy(pmi[vocab.index(word2), :])
vec2[idx] = 0
print np.dot(unitvec(vec1), unitvec(vec2))

plt.figure()
plt.plot(unitvec(vec1[:500]))
plt.plot(unitvec(vec2[:500]))
##

##
D = cofreq.sum()
freq = cofreq.sum(axis=1)
C = freq[int(0.5*len(freq))]

widx = vocab.index('prime')

def wPMI2PMI(inner, C, D, freq, widx,iter=3):
    output = np.log(D/np.maximum(freq[widx], freq))
    tmp = inner * C / freq[widx] * D / freq
    for _ in xrange(iter):
        output -= (output * np.log(output) - tmp)/(1 + np.log(output))
    return np.log(output)

widx = vocab.index('prime')
inner = pmi[widx, :] * cofreq[widx, :] / C
foo = wPMI2PMI(inner, C, D, freq, widx, iter=4)
plt.plot(foo[:500])

plt.figure()
plt.plot(pmi[widx,:500])



##


plt.figure()
foo = []
for i in np.arange(1,20,0.5):
    vec1 = pmi[vocab.index(word1), :] * 1.0/(1+np.exp(cofreq[vocab.index(word1)] * -i))
    vec2 = pmi[vocab.index(word2), :] * 1.0/(1+np.exp(cofreq[vocab.index(word2)] * -i))
    foo.append(np.dot(unitvec(vec1), unitvec(vec2)))
plt.plot(foo)



plt.figure()
plt.plot(np.dot((pmi[vocab.index(word1)]/vecnorm[vocab.index(word1)])[np.newaxis, :], pmi/(0.0001+vecnorm[np.newaxis, :])).squeeze()[:500])


print [w for w in vocab[np.argsort(np.dot(vec1/norm1, wpmi/np.norm(wpmi, axis=1)))]]
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
