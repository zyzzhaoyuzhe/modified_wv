import logging, sys
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from wikicorpus import WikiCorpus

folder = sys.argv[1]
nprocess = None
if len(sys.argv) > 2:
    nprocess = sys.argv[2]
fpre = folder + "enwiki-20160920-pages-articles-multistream.xml.bz2"
fout = folder + "enwiki-20160920"

wiki = WikiCorpus(fpre, lemmatize=True, dictionary=False, processes=nprocess)
gen = wiki.get_texts()

with open(fout, 'wb') as h:
    for sents in gen:
        for sent in sents:
            h.write(' '.join(sent) + '\n')