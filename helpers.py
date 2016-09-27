from __future__ import division
import numpy as np
import nltk, re, os, time
from nltk.corpus import wordnet, stopwords
from six import string_types


def sigma(x):
    return 1 / (1 + np.exp(-x))


def inner2prob(count_c, count_w, D, inner, niter=3):
    """
    Use newtown method to convert embedding vectors inner product to joint probability.
    :param count_c: input of the network that is always a scalar.
    :param count_w: output of the network that outputs positive contexts and negtive contexts. An array.
    :param D:
    :param inner:
    :param niter:
    :return:
    """
    # print count_c, count_w, D, inner
    count_w = np.asarray(count_w)
    jcount_m, m, M = inner_minmax(count_c, count_w, D)
    output = np.zeros(jcount_m.shape)
    idx1 = inner < m
    output[idx1] = jcount_m[idx1]
    idx2 = inner > M
    output[idx2] = np.minimum(count_c, count_w)[idx2]
    idx = ~(idx1 | idx2)
    if idx.sum():
        pp = count_c * np.asarray(count_w[idx])

        def func(x):
            return x - (x * np.log(x * D / pp) - np.asarray(inner[idx]) * D) / (np.log(x * D / pp) + 1)

        foo = np.minimum(count_c, count_w[idx])
        for i in range(niter):
            foo = func(foo)
        output[idx] = foo
    return output.astype(int)+1


def inner_minmax(count_c, count_w, D):
    count_c = np.asarray(count_c)
    count_w = np.asarray(count_w)

    def func(x):
        return x / D * np.log(x * D / count_c / count_w)

    foo = (count_c * count_w) / D * np.exp(-1)
    return (foo, func(foo), func(np.minimum(count_c, count_w)))


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


def pre_process(content, lower=True, tokenize=True, lemmatization=True):
    """
    Text preprocess includes lowercase, tokenize and lemmatization.
    :param content: If it is string type, the content is a sent; if it is a list,
    :param lower:
    :param tokenize:
    :param Lemmatization:
    :return:
    """
    lmtz = nltk.stem.WordNetLemmatizer()
    if isinstance(content, string_types):  # content is a sent
        # convert to lower case
        if lower:
            sent = content.lower()
        # word tokenize
        sent = sent.replace('-', ' ')
        if tokenize:
            tokens = nltk.word_tokenize(sent)
            tags = nltk.pos_tag(tokens)
            foo = []
            for tag in tags:
                token = tag[0]
                treebank_tag = tag[1]
                if lemmatization:
                    token = lmtz.lemmatize(token, get_wordnet_pos(treebank_tag)).encode('utf8')
                if re.findall(r'[a-z]', token) and token != "'s":
                    foo.append(token)
        return foo
    else:  # content is a list of words
        tags = nltk.pos_tag(content)
        foo = []
        for tag in tags:
            token = tag[0]
            treebank_tag = tag[1]
            if lower:
                token = token.lower()
            if lemmatization:
                token = lmtz.lemmatize(token, get_wordnet_pos(treebank_tag)).encode('utf8')
            if re.findall(r'[a-z]', token) and token != "'s":
                foo.append(token)
        return foo


def keep_vocab_item(word, count, min_count, trim_rule=None):
    default_res = count >= min_count

    if trim_rule is None:
        return default_res
    else:
        rule_res = trim_rule(word, count, min_count)
        if rule_res == RULE_KEEP:
            return True
        elif rule_res == RULE_DISCARD:
            return False
        else:
            return default_res


class RepeatCorpusNTimes():
    def __init__(self, corpus, n):
        """
        Repeat a `corpus` `n` times.

        >>> corpus = [[(1, 0.5)], []]
        >>> list(RepeatCorpusNTimes(corpus, 3)) # repeat 3 times
        [[(1, 0.5)], [], [(1, 0.5)], [], [(1, 0.5)], []]
        """
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        for _ in xrange(self.n):
            for document in self.corpus:
                yield document


def qsize(queue):
    """Return the (approximate) queue size where available; -1 where not (OS X)."""
    try:
        return queue.qsize()
    except NotImplementedError:
        # OS X doesn't support qsize
        return -1
