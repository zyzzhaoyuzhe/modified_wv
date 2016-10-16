import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import mword2vec
from helpers import smartfile
import sys
import multiprocessing
import argparse
import time
import gensim

parser = argparse.ArgumentParser(description="Train word vec on wiki data")

parser.add_argument('folder')
parser.add_argument('--size', type=int, default=300)
parser.add_argument('--min_count', type=int, default=5)
parser.add_argument('--sample', type=float, default=0)
parser.add_argument('--wPMI', type=int, default=1)
parser.add_argument('--smooth_power', type=float, default=0)
parser.add_argument('--negative', type=int, default=5)
parser.add_argument('--neg_mean', type=int, default=1)
parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count())


args = parser.parse_args()

text = smartfile(args.folder + 'enwiki-20160920')

print "Train on wiki dump data with the parameters"
print args

for i in range(5):
    print "Start in {0} seconds".format(5-i)
    time.sleep(1)

# model = mword2vec.mWord2Vec(text, size=args.size, min_count=args.min_count, sample=args.sample, wPMI=args.wPMI,
#                             smooth_power=args.smooth_power, negative=args.negative, neg_mean=args.neg_mean, workers=args.workers)
model = gensim.models.Word2Vec(text, min_count=1, sample=0, sg=1, negative=5, workers=1)