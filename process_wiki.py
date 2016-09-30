import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from wikicorpus import WikiCorpus

folder = "/media/vincent/Data/Dataset/wiki_en/"
fpre = folder + "enwiki-20160920-pages-articles-multistream.xml.bz2"
fout = folder + "enwiki-20160920"

wiki = WikiCorpus(fpre, lemmatize=True, dictionary=False, processes=4)
gen = wiki.get_texts()

with open(fout, 'wb') as h:
    for sents in gen:
        for sent in sents:
            h.write(' '.join(sent) + '\n')