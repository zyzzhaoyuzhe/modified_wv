# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import nltk, re, os, time
from nltk.corpus import wordnet, stopwords
from six import string_types
import nltk


def sigma(x):
    return 1 / (1 + np.exp(-x))


def inner2prob(count_c, count_w, D, C, inner, niter=3):
    """

    :param count_c:
    :param count_w:
    :param D:
    :param C:
    :param inner:
    :param niter:
    :return:
    """
    # print count_c, count_w, D, inner
    count_w = np.asarray(count_w)
    jcount_m, m, M = inner_minmax(count_c, count_w, D, C)
    output = np.zeros(jcount_m.shape)
    idx1 = inner < m
    output[idx1] = jcount_m[idx1]
    idx2 = inner > M
    output[idx2] = np.minimum(count_c, count_w)[idx2]
    idx = ~(idx1 | idx2)
    if idx.sum():
        pp = count_c * np.asarray(count_w[idx])

        def func(x):
            return x - (x * np.log(x * D / pp) - np.asarray(inner[idx]) * D / C) / (np.log(x * D / pp) + 1)

        foo = np.minimum(count_c, count_w[idx])
        for i in range(niter):
            foo = func(foo)
        output[idx] = foo
    return output.astype(int)+1


def inner_minmax(count_c, count_w, D, C):
    count_c = np.asarray(count_c)
    count_w = np.asarray(count_w)

    def func(x):
        # joint counts to inner product
        return x * C / D * np.log(x * D / count_c / count_w)

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
        return wordnet.NOUN


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


def tokenize(text, lowercase=False, deacc=False, errors="strict", to_lower=False, lower=False):
    """
    Iteratively yield tokens as unicode strings, removing accent marks
    and optionally lowercasing the unidoce string by assigning True
    to one of the parameters, lowercase, to_lower, or lower.

    Input text may be either unicode or utf8-encoded byte string.

    The tokens on output are maximal contiguous sequences of alphabetic
    characters (no digits!).

    >>> list(tokenize('Nic nemůže letět rychlostí vyšší, než 300 tisíc kilometrů za sekundu!', deacc = True))
    [u'Nic', u'nemuze', u'letet', u'rychlosti', u'vyssi', u'nez', u'tisic', u'kilometru', u'za', u'sekundu']

    """
    lowercase = lowercase or to_lower or lower
    text = to_unicode(text, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    sents = nltk.sent_tokenize(text)
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        yield tokens


def lemmatize(content, allowed_tags=re.compile('(NN|VB|JJ|RB)'),
              light=False, stopwords=frozenset(), min_length=2, max_length=15):
    """
    This function is only available when the optional 'pattern' package is installed.

    Use the English lemmatizer from `pattern` to extract UTF8-encoded tokens in
    their base form=lemma, e.g. "are, is, being" -> "be" etc.
    This is a smarter version of stemming, taking word context into account.

    Only considers nouns, verbs, adjectives and adverbs by default (=all other lemmas are discarded).

    >>> lemmatize('Hello World! How is it going?! Nonexistentword, 21')
    ['world/NN', 'be/VB', 'go/VB', 'nonexistentword/NN']

    >>> lemmatize('The study ranks high.')
    ['study/NN', 'rank/VB', 'high/JJ']

    >>> lemmatize('The ranks study hard.')
    ['rank/NN', 'study/VB', 'hard/RB']

    """

    # if not has_pattern():
    #     raise ImportError("Pattern library is not installed. Pattern library is needed in order  \
    #      to use lemmatize function")
    # from pattern.en import parse

    # if light:
    #     import warnings
    #
    #     warnings.warn("The light flag is no longer supported by pattern.")

    # tokenization in `pattern` is weird; it gets thrown off by non-letters,
    # producing '==relate/VBN' or '**/NN'... try to preprocess the text a little
    # FIXME this throws away all fancy parsing cues, including sentence structure,
    # abbreviations etc.
    lmtz = nltk.stem.WordNetLemmatizer()
    result = []
    for sent in tokenize(content, lower=True, errors='ignore'):
        foo = []
        tags = nltk.pos_tag(sent)
        for token, treebank_tag in tags:
            lemma = lmtz.lemmatize(token, get_wordnet_pos(treebank_tag).encode('utf8'))
            if min_length <= len(lemma) <= max_length and not lemma.startswith(('_', '-', "'", '='))\
                    and not lemma.endswith(('_', '=')) and lemma not in stopwords:
                if allowed_tags.match(treebank_tag):
                    lemma += "/" + treebank_tag
                    foo.append(lemma.encode('utf8'))
        result.append(foo)
    return result

