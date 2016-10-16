import logging, sys
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from wikicorpus import WikiCorpus

folder = '/media/vincent/Data/Dataset/wiki_en/'
nprocess = None
# if len(sys.argv) > 2:
#     nprocess = int(sys.argv[2])
nprocess = 4
fpre = folder + "enwiki-20160920-pages-articles-multistream.xml"
fout = folder + "enwiki-20160920_basic"

wiki = WikiCorpus(fpre, lemmatize=True, dictionary=False, processes=4)
gen = wiki.get_texts()

count = 1
with open(fout, 'w') as h:
    for sents in gen:
        for sent in sents:
            h.write(' '.join(sent) + '\n')