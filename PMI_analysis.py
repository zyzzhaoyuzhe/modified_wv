import PMI
import cPickle as pickle

data = pickle.load(open('ap.p','r'))
vocab_all, freq_all = PMI.getvocab(data)
vocab, freq = PMI.getvocab(data)
cofreq = PMI.get_cooccurrance(data, vocab_all, window=5, dynamic_window=True)

pmi = PMI.get_pmi(cofreq)
wpmi = PMI.get_wpmi(cofreq, pmi, type='wpmi')

# 'wpmi' 'ppwpmi' 'minwpmi'



data_stop = clear_wordset(data, set(stopwords.words('english')))
vocab, freq = getvocab(data_stop)
cofreq = get_cooccurrance(data_stop, vocab, window=5, dynamic_window=True)

data_lowfreq_stop = clear_wordset(data, set(vocab_all[-17000:] + stopwords.words('english')))
vocab, freq = getvocab(data_lowfreq_stop)
cofreq = get_cooccurrance(data_lowfreq_stop, vocab, window=5, dynamic_window=True)

data_lowfreq = clear_wordset(data, set(vocab_all[-17000:]))
vocab, freq = getvocab(data_lowfreq)
cofreq = get_cooccurrance(data_lowfreq, vocab, window=5, dynamic_window=True)

n_largest(np.tril(wpmi, -1), vocab, 20)
n_largest(np.tril(pmi2, -1), vocab, 20)

show_context(wpmi, vocab, 'vote', 10)
show_context(pmi2, vocab, 'prime', 10)