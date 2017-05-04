import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
import operator
import math
from helpers import smartfile

class Bag_of_ngrams(object):
    def __init__(self, ngram, maxmem=10000000):
        self.n = ngram
        self.unigram = defaultdict(int)
        self.ngram = defaultdict(int)
        self.total = [0] * ngram
        self.maxmem = maxmem

    def prune(self, dic, thres):
        firsttimmer = 1
        if len(dic) > self.maxmem:
            while len(dic) > 0.5*self.maxmem:
                for word in list(dic):
                    if dic[word] < thres:
                        del dic[word]
                if not firsttimmer:
                    thres += 1
                firsttimmer = 0
        return thres

    def count(self, text, per=10000, dic={}):
        thres_uni = 1
        thres_ngram = 1
        if not dic:
            dic = {}
        for l, sent in enumerate(text):
            if l % per == 0:
                logging.info('Finish {} lines; unigram {} @ threshold {}; ngram {} @ threshold {}'.format(l, len(self.unigram), thres_uni, len(self.ngram), thres_ngram))
            for w in sent:
                self.unigram[w] += 1
                self.total[0] += 1
            i = 0
            while i < len(sent):
                for j in range(min(len(sent), i + self.n) - 1, i, -1):
                    word = ' '.join(sent[i:j + 1])
                    self.ngram[word] += 1
                    self.total[j - i] += 1
                    if word in dic:
                        i = j
                        break
                i += 1
            #prune
            thres_uni = self.prune(self.unigram, thres_uni)
            thres_ngram = self.prune(self.ngram, thres_ngram)

    def process(self, text, topN):
        self.count(text)
        logging.info('start sorting')
        self.ngram = sorted(self.ngram.items(), key=operator.itemgetter(1), reverse=True)[:topN]

class Trie(object):
    def __init__(self):
        self.tree = {}

    def add(self, word):
        if not isinstance(word, basestring):
            return
        word = word.split()
        node = self.tree
        for l in word:
            if l not in node:
                node[l] = {}
            node = node[l]

    def __contains__(self, word):
        node = self.tree
        word = word.split()
        for l in word:
            if l in node:
                node = node[l]
            else:
                return False
        return True

class Wpmi_bag_of_ngrams(Bag_of_ngrams):
    def get_ngram_wpmi(self, topN):
        logging.info('Start get_ngram_wpmi')
        ngram_wpmi = defaultdict(int)
        for key, value in self.ngram.items():
            ch = key.split()
            l = len(ch)
            ngram_wpmi[key] = value * (math.log(value) -
                                       math.log(self.total[l - 1]) -
                                     sum(math.log(self.unigram[w]) for w in ch) +
                                     l * math.log(self.total[0]))
        ngram_wpmi = sorted(ngram_wpmi.items(), key=operator.itemgetter(1), reverse=True)[:topN]
        self.ngram_wpmi = ngram_wpmi

    def get_dic(self):
        # TODO: use prefix tree
        logging.info('Start get_dic')
        dic = set()
        substr = Trie()
        for w, _ in self.ngram_wpmi:
            if w in substr:
                continue
            dic.add(w)
            w = w.split()
            for i in range(len(w)-1):
                substr.add(' '.join(w[i:]))
        return dic

        # O(n^2) slow implementation
        # output = set()
        # for w, _ in self.ngram_wpmi:
        #     if any(w in key for key in output):
        #         continue
        #     output.add(w)
        # return output

    def process(self, text, topN, iter=2, dic={}):
        for i in range(iter):
            self.unigram = defaultdict(int)
            self.ngram = defaultdict(int)
            self.total = [0] * self.n
            self.count(text, dic=dic)
            self.get_ngram_wpmi(topN)
            logging.info('start sorting')
            self.ngram = sorted(self.ngram.items(), key=operator.itemgetter(1), reverse=True)[:topN]
            if i == iter - 1:
                break
            dic = self.get_dic()



if __name__ == '__main__':
    text = smartfile('/media/vincent/Data-adhoc/wiki_dumps/wiki_en/enwiki-basic')
    obj = Wpmi_bag_of_ngrams(5, maxmem=10000000)
    obj.process(text, 1000000, iter=1)

    with open('enwiki-basic-{}gram-freq-list'.format(5), 'w') as h:
        for idx in xrange(len(obj.ngram)):
            h.write('{}\t{}\t{}\n'.format(idx, obj.ngram[idx][0].encode('utf-8'), obj.ngram[idx][1]))

    dic = obj.get_dic()
    obj.process(text, 1000000, iter=1, dic=dic)

    with open('enwiki-basic-{}gram-wpmi-list-iter{}'.format(5, 2), 'w') as h:
        for idx in xrange(len(obj.ngram_wpmi)):
            h.write('{}\t{}\t{}\n'.format(idx,
                                          obj.ngram_wpmi[idx][0].encode('utf-8'),
                                          math.log(obj.ngram_wpmi[idx][1]) if obj.ngram_wpmi[idx][1] > 0 else 0))







