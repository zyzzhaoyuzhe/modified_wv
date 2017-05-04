import logging
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from helpers import smartfile
import lib_ngrams
import math


class feature(object):
    def __init__(self, file):
        self.label = []
        self.data = []

class BOW_count(feature):
    def load(self):

def load_zhang(file):
    h = open(file, 'r')
    for line in h:
        line = line.split('","',2)
        label = int(line[0][1:])
        title = line[1]
        body = line[2][:-3]
        yield label, title, body