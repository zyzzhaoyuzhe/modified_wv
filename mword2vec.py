"""
A python implementation of modified word2vec (skip gram with negative sampling based on weighted PMI) which builds on gensim.
"""
from __future__ import division
import logging
import numpy as np
from numpy import float32 as REAL
import threading
from Queue import Queue, Empty, PriorityQueue
from types import GeneratorType
from collections import defaultdict
from copy import deepcopy
from six import iteritems, itervalues, string_types
from six.moves import xrange
from timeit import default_timer
from gensim import utils, matutils
import helpers

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(0)

MAX_WORDS_IN_BATCH = 10000
try:
    from mword2vec_inner import train_batch_sg
    from mword2vec_inner import FAST_VERSION
except ImportError:
    FAST_VERSION = -1


    def train_sg_pair(model, word, context_index, alpha, learn_vectors=True, learn_hidden=True,
                      context_vectors=None, context_locks=None):
        if context_vectors is None:
            context_vectors = model.syn0
        if context_locks is None:
            context_locks = model.syn0_lockf

        if word not in model.vocab:
            return
        predict_word = model.vocab[word]  # target word (NN output)

        l1 = context_vectors[context_index]  # input word (NN input/projection layer)
        lock_factor = context_locks[context_index]

        neu1e = np.zeros(l1.shape)

        if model.negative:
            # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
            word_indices = [predict_word.index]
            while len(word_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
                if w != predict_word.index:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
            if model.wPMI:
                inner = np.dot(l1, l2b.T)
                # get counts of contexts
                count_context = model.vocab[model.index2word[context_index]].count
                counts_word = [model.vocab[model.index2word[idx]].count for idx in word_indices]
                # joint counts of (w,c) from <w,c>
                C = 344622
                D = model.words_cumnum
                # C = len(model.vocab) ** 2
                # D = model.cum_table[-1]
                jcounts = helpers.inner2prob(count_context, counts_word, D, C, inner)
                weight = C / D * jcounts
                # For debug
                logger.debug(
                    "Show the max joint prob: " + " ".join(str(e) for e in np.minimum(count_context, counts_word)))
                logger.debug("Show the joint prob: " + " ".join(str(e) for e in jcounts))
                logger.debug("show the inner prod: " + " ".join(str(e) for e in inner))
                logger.debug("Temperature: " + " ".join(str(e) for e in weight))
                # calculate gradient
                fb = 1. / (1. + np.exp(-1 / weight * inner))  # propagate hidden -> output
                gb = (model.neg_labels - fb) * alpha / weight
                if model.neg_mean:
                    # use mean for negative sampling
                    gb = gb * model.neg_mean_weight
            else:
                fb = 1. / (1. + np.exp(-np.dot(l1, l2b.T)))  # propagate hidden -> output
                gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate

            logger.debug("Show the gradient: " + " ".join(str(e) for e in gb))

            if learn_hidden:
                model.syn1neg[word_indices] += np.outer(gb, l1)  # learn hidden -> output
            neu1e += np.dot(gb, l2b)  # save error

        if learn_vectors:
            l1 += neu1e * lock_factor  # learn input -> hidden (mutates model.syn0[word2.index], if that is l1)
        return neu1e


    def train_batch_sg(model, sentences, alpha, work=None):
        """
        Update skip-gram model by training on a sequence of sentences.

        Each sentence is a list of string tokens, which are looked up in the model's
        vocab dictionary. Called internally from `Word2Vec.train()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.

        """
        result = 0
        for sentence in sentences:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2 ** 32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    # don't train on the `word` itself
                    if pos2 != pos:
                        train_sg_pair(model, model.index2word[word.index], word2.index, alpha)
            result += len(word_vocabs)
        return result


    def score_sg_pair(model, word, word2):
        l1 = model.syn0[word2.index]
        l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
        sgn = (-1.0) ** word.code  # ch function, 0-> 1, 1 -> -1
        lprob = -np.log(1.0 + np.exp(-sgn * np.dot(l1, l2a.T)))
        return sum(lprob)


    def score_sentence_sg(model, sentence, work=None):
        """
        Obtain likelihood score for a single sentence in a fitted skip-gram representaion.

        The sentence is a list of Vocab objects (or None, when the corresponding
        word is not in the vocabulary). Called internally from `Word2Vec.score()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.

        """

        log_prob_sentence = 0.0
        if model.negative:
            raise RuntimeError("scoring is only available for HS=True")

        word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab]
        for pos, word in enumerate(word_vocabs):
            if word is None:
                continue  # OOV word in the input sentence => skip

            # now go over all words from the window, predicting each one in turn
            start = max(0, pos - model.window)
            for pos2, word2 in enumerate(word_vocabs[start: pos + model.window + 1], start):
                # don't train on OOV words and on the `word` itself
                if word2 is not None and pos2 != pos:
                    log_prob_sentence += score_sg_pair(model, word, word2)

        return log_prob_sentence


class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class mWord2Vec(utils.SaveLoad):
    """
    A modified skip gram model with negative sampling for word embedding.
    """

    def __init__(
            self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            negative=5, smooth_power=0.75, neg_mean=0, hashfxn=hash, epoch=5, null_word=0, wPMI=0,
            trim_rule=None, sorted_vocab=1, init="uniform", batch_words=MAX_WORDS_IN_BATCH):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        :param sentences:

        :param size:

        :param alpha: the initial learning rate (will linearly drop to `min_alpha` as training progresses).

        :param window: the maximum distance between the current and predicted word within a sentence.

        :param min_count: ignore all words with total frequency lower than this.

        :param max_vocab_size: limit RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent np.ones. Every 10 million word types
        need about 1GB of RAM. Set to `None` for no limit (default).

        :param sample: threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, useful range is (0, 1e-5).

        :param seed: for the random number generator. Initial vectors for each
        word are seeded with a hash of the concatenation of word + str(seed).
        Note that for a fully deterministically-reproducible run, you must also limit the model to
        a single worker thread, to eliminate ordering jitter from OS thread scheduling. (In Python
        3, reproducibility between interpreter launches also requires use of the PYTHONHASHSEED
        environment variable to control hash randomization.)

        :param workers: use this many worker threads to train the model (=faster training with multicore machines).

        :param min_alpha:

        :param negative: if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).
        Default is 5. If set to 0, no negative samping is used.

        :param neg_mean:

        :param hashfxn: hash function to use to randomly initialize weights, for increased
        training reproducibility. Default is Python's rudimentary built in hash function.

        :param epoch: number of iterations (epochs) over the corpus. Default is 5.

        :param null_word:

        :param wPMI:

        :param trim_rule: vocabulary trimming rule, specifies whether certain words should remain
        in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).
        Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and
        returns either `helpers.RULE_DISCARD`, `helpers.RULE_KEEP` or `helpers.RULE_DEFAULT`.
        Note: The rule, if given, is only used prune vocabulary during build_vocab() and is not stored as part
        of the model.
        :param sorted_vocab:
        :param batch_words: target size (in words) for batches of examples passed to worker threads (and
        thus cython routines). Default is 10000. (Larger batches will be passed if individual
        texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        """

        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.vector_size = int(size)
        self.layer1_size = int(size)
        self.weight_init = init
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")

        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.min_alpha = float(min_alpha)

        self.window = int(window)

        self.seed = seed
        self.random = np.random.RandomState(seed)

        self.sample = sample  # For downsampling
        self.workers = int(workers)
        self.iter = epoch

        self.hashfxn = hashfxn

        self.null_word = null_word
        self.train_count = 0
        self.total_train_time = 0
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

        self.negative = negative
        self.neg_mean = neg_mean
        self.cum_table = None  # for negative sampling
        self.wPMI = wPMI
        self.smooth_power = smooth_power

        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            # self.build_vocab(sentences, trim_rule=trim_rule)
            # self.train(sentences)

    def make_cum_table(self, power=0.75, domain=2 ** 31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or np.ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.

        Called internally from 'build_vocab()'.
        """
        vocab_size = len(self.index2word)
        self.cum_table = np.zeros(vocab_size, dtype=np.uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.vocab[word].count ** power for word in self.vocab]))
        cumulative = 0.0
        for word_index in range(vocab_size):
            cumulative += self.vocab[self.index2word[word_index]].count ** power / train_words_pow
            self.cum_table[word_index] = round(cumulative * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain
        # calculate total number of words
        self.words_cumnum = train_words_pow

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.scan_vocab(sentences, progress_per=progress_per, trim_rule=trim_rule)  # initial survey
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab,
                         trim_rule=trim_rule)  # trim by min_count & precalculate downsampling
        self.finalize_vocab()  # build tables & arrays

    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None):
        """Do an initial scan of all words appearing in sentences."""
        logger.info("collecting all words and their counts")
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        checked_string_types = 0
        for sentence_no, sentence in enumerate(sentences):
            if not checked_string_types:
                if isinstance(sentence, string_types):
                    logger.warn("Each 'sentences' item should be a list of words (usually unicode strings)."
                                "First item here is instead plain %s.", type(sentence))
                checked_string_types += 1
            if sentence_no % progress_per == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                            sentence_no, sum(itervalues(vocab)) + total_words, len(vocab))
            for word in sentence:
                vocab[word] += 1

            if self.max_vocab_size and len(vocab) > 2*self.max_vocab_size:
                total_words += utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 0.5
        # reduce vocabsize to max_vocab_size
        Q = PriorityQueue(maxsize=self.max_vocab_size)
        for val in vocab.itervalues():
            if Q.full():
                foo = Q.get()
                if val > foo:
                    Q.put_nowait(val)
                else:
                    Q.put_nowait(foo)
            else:
                Q.put_nowait(val)
        utils.prune_vocab(vocab, Q.get_nowait()+1, trim_rule=trim_rule)

        total_words += sum(itervalues(vocab))
        logger.info("collected %i word types from a corpus of %i raw words and %i sentences",
                    len(vocab), total_words, sentence_no + 1)
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab

    def scale_vocab(self, min_count=None, sample=None, dry_run=False, keep_raw_vocab=False, trim_rule=None):
        """
        Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).

        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.

        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.

        """
        min_count = min_count or self.min_count
        sample = sample or self.sample

        # Discard words less-frequent than min_count
        if not dry_run:
            self.index2word = []
            # make stored settings match these applied settings
            self.min_count = min_count
            self.sample = sample
            self.vocab = {}
        drop_unique, drop_total, retain_total, original_total = 0, 0, 0, 0
        retain_words = []
        for word, v in iteritems(self.raw_vocab):
            if utils.keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                retain_words.append(word)
                retain_total += v
                original_total += v
                if not dry_run:
                    self.vocab[word] = Vocab(count=v, index=len(self.index2word))
                    self.index2word.append(word)
            else:
                drop_unique += 1
                drop_total += v
                original_total += v
        logger.info("min_count=%d retains %i unique words (drops %i)",
                    min_count, len(retain_words), drop_unique)
        logger.info("min_count leaves %i word corpus (%i%% of original %i)",
                    retain_total, retain_total * 100 / max(original_total, 1), original_total)

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + np.sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (np.sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.vocab[w].sample_int = int(round(word_probability * 2 ** 32))

        if not dry_run and not keep_raw_vocab:
            logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
                    downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total)

        # return from each step: words-affected, resulting-corpus-size
        report_values = {'drop_unique': drop_unique, 'retain_total': retain_total,
                         'downsample_unique': downsample_unique, 'downsample_total': int(downsample_total)}

        # print extra memory estimates
        report_values['memory'] = self.estimate_memory(vocab_size=len(retain_words))

        return report_values

    def finalize_vocab(self):
        """Build tables and model weights based on final vocabulary settings."""
        if not self.index2word:
            self.scale_vocab()
        if self.sorted_vocab:
            self.sort_vocab()
        # if self.hs:
        #     # add info about each word's Huffman encoding
        #     self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table(power=self.smooth_power)
        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input, never predicted, so count, huffman-point, etc doesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(self.vocab)
            self.index2word.append(word)
            self.vocab[word] = v
        # set initial input/projection and hidden weights
        self.reset_weights()

    def sort_vocab(self):
        """Sort the vocabulary so the most frequent words have the lowest indexes."""
        if hasattr(self, 'syn0'):
            raise RuntimeError("must sort before initializing vectors/weights")
        self.index2word.sort(key=lambda word: self.vocab[word].count, reverse=True)
        for i, word in enumerate(self.index2word):
            self.vocab[word].index = i

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings and provided vocabulary size."""
        vocab_size = vocab_size or len(self.vocab)
        report = report or {}
        # report['vocab'] = vocab_size * (700 if self.hs else 500)
        report['vocab'] = vocab_size * 500
        report['syn0'] = vocab_size * self.vector_size * np.dtype(REAL).itemsize
        if self.negative:
            report['syn1neg'] = vocab_size * self.layer1_size * np.dtype(REAL).itemsize
        report['total'] = sum(report.values())
        logger.info("estimated required memory for %i words and %i dimensions: %i bytes",
                    vocab_size, self.vector_size, report['total'])
        return report

    def _do_train_job(self, sentences, alpha, inits):
        """
        Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.
        """
        # work, neu1 = inits
        work = inits
        tally = 0
        tally += train_batch_sg(self, sentences, alpha, work)
        return tally, self._raw_word_count(sentences)

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence) for sentence in job)

    def train(self, sentences, total_words=None, word_count=0,
              total_examples=None, queue_factor=2, report_delay=1.0):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, either total_examples
        (count of sentences) or total_words (count of raw words in sentences) should be provided, unless the
        sentences are the same as those that were used to initially build the vocabulary.

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for Word2Vec, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = np.zeros(self.negative + 1)
                self.neg_labels[0] = 1.
            self.neg_mean_weight = []
            if self.neg_mean:
                self.neg_mean_weight = np.ones(self.negative + 1) * 1 / self.negative
                self.neg_mean_weight[0] = 1

        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using wPMI=%s neg_mean=%s sample=%s negative=%s",
            self.workers, len(self.vocab), self.layer1_size, self.wPMI, self.neg_mean, self.sample, self.negative)
        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        if not hasattr(self, 'syn0'):
            raise RuntimeError("you must first finalize vocabulary before training the model")

        if total_words is None and total_examples is None:
            if self.corpus_count:
                total_examples = self.corpus_count
                logger.info("expecting %i sentences, matching count from corpus used for vocabulary survey",
                            total_examples)
            else:
                raise ValueError(
                    "you must provide either total_words or total_examples, to enable alpha and progress calculations")

        job_tally = 0

        if self.iter > 1:
            # Create an iterator that repeats sentences self.iter times
            sentences = utils.RepeatCorpusNTimes(sentences, self.iter)
            total_words = total_words and total_words * self.iter
            total_examples = total_examples and total_examples * self.iter

        def worker_loop():
            """Train the model, lifting lists of sentences from the job_queue."""
            work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            # neu1 = mathhelpers.zeros_aligned(self.layer1_size, dtype=REAL)
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                sentences, alpha = job
                tally, raw_tally = self._do_train_job(sentences, alpha, work)
                progress_queue.put((len(sentences), tally, raw_tally))  # report back progress
                jobs_processed += 1
            logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_alpha = self.alpha
            if next_alpha > self.min_alpha_yet_reached:
                logger.warn("Effective 'alpha' higher than previous training cycles")
            self.min_alpha_yet_reached = next_alpha
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):
                sentence_length = self._raw_word_count([sentence])

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    logger.debug(
                        "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                        job_no, batch_size, len(job_batch), next_alpha)
                    job_no += 1
                    job_queue.put((job_batch, next_alpha))

                    # update the learning rate for the next job
                    if self.min_alpha < next_alpha:
                        if total_examples:
                            # examples-based decay
                            pushed_examples += len(job_batch)
                            progress = 1.0 * pushed_examples / total_examples
                        else:
                            # words-based decay
                            pushed_words += self._raw_word_count(job_batch)
                            progress = 1.0 * pushed_words / total_words
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * progress
                        next_alpha = max(self.min_alpha, next_alpha)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
                logger.debug(
                    "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                    job_no, batch_size, len(job_batch), next_alpha)
                job_no += 1
                job_queue.put((job_batch, next_alpha))

            if job_no == 0 and self.train_count == 0:
                logger.warning(
                    "train() called with an empty iterator (if not intended, "
                    "be sure to provide a corpus that offers restartable "
                    "iteration = an iterable)."
                )

            # give the workers heads up that they can finish -- no more work!
            for _ in xrange(self.workers):
                job_queue.put(None)
            logger.debug("job loop exiting, total %i jobs", job_no)

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        ## debug
        # workers = [threading.Thread(target=job_producer)]

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        example_count, trained_word_count, raw_word_count = 0, 0, word_count
        start, next_report = default_timer() - 0.00001, 1.0

        while unfinished_worker_count > 0:

            ## debug
            # sentences, alpha = job_queue.get()
            # tally, raw_tally = self._do_train_job(sentences, alpha, (None, None))

            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                if total_examples:
                    # examples-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i, vec_norm %.2f",
                        100.0 * example_count / total_examples, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue), np.linalg.norm(self.syn0[1]))
                else:
                    # words-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed)
        if job_tally < 10 * self.workers:
            logger.warn("under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay")

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warn("supplied example count (%i) did not equal expected count (%i)", example_count, total_examples)
        if total_words and total_words != raw_word_count:
            logger.warn("supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words)

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self.clear_sims()
        return trained_word_count

    # basics copied from the train() function
    def score(self, sentences, total_sentences=int(1e6), chunksize=100, queue_factor=2, report_delay=1):
        """
        Score the log probability for a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.
        This does not change the fitted model in any way (see Word2Vec.train() for that).

        We have currently only implemented score for the hierarchical softmax scheme,
        so you need to have run word2vec with hs=1 and negative=0 for this to work.

        Note that you should specify total_sentences; we'll run into problems if you ask to
        score more than this number of sentences but it is inefficient to set the value too high.

        See the article by [taddy]_ and the gensim demo at [deepir]_ for examples of how to use such scores in document classification.

        .. [taddy] Taddy, Matt.  Document Classification by Inversion of Distributed Language Representations, in Proceedings of the 2015 Conference of the Association of Computational Linguistics.
        .. [deepir] https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/deepir.ipynb

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension compilation failed, scoring will be slow. "
                          "Install a C compiler and reinstall gensim for fastness.")

        logger.info(
            "scoring sentences with %i workers on %i vocabulary and %i features, "
            "using wPMI=%s neg_mean=%s sample=%s and negative=%s",
            self.workers, len(self.vocab), self.layer1_size, self.wPMI, self.neg_mean, self.sample, self.negative)

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before scoring new data")

        # if not self.hs:
        #     raise RuntimeError("we have only implemented score for hs")

        def worker_loop():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = np.zeros(1, dtype=REAL)  # for sg hs, we actually only need one memory loc (running sum)
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            while True:
                job = job_queue.get()
                if job is None:  # signal to finish
                    break
                ns = 0
                for sentence_id, sentence in job:
                    if sentence_id >= total_sentences:
                        break
                    score = score_sentence_sg(self, sentence, work)
                    sentence_scores[sentence_id] = score
                    ns += 1
                progress_queue.put(ns)  # report progress

        start, next_report = default_timer(), 1.0
        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        sentence_count = 0
        sentence_scores = utils.zeros_aligned(total_sentences, dtype=REAL)

        push_done = False
        done_jobs = 0
        jobs_source = enumerate(helpers.grouper(enumerate(sentences), chunksize))

        # fill jobs queue with (id, sentence) job items
        while True:
            try:
                job_no, items = next(jobs_source)
                if (job_no - 1) * chunksize > total_sentences:
                    logger.warning(
                        "terminating after %i sentences (set higher total_sentences if you want more).",
                        total_sentences)
                    job_no -= 1
                    raise StopIteration()
                logger.debug("putting job #%i in the queue", job_no)
                job_queue.put(items)
            except StopIteration:
                logger.info(
                    "reached end of input; waiting to finish %i outstanding jobs",
                    job_no - done_jobs + 1)
                for _ in xrange(self.workers):
                    job_queue.put(None)  # give the workers heads up that they can finish -- no more work!
                push_done = True
            try:
                while done_jobs < (job_no + 1) or not push_done:
                    ns = progress_queue.get(push_done)  # only block after all jobs pushed
                    sentence_count += ns
                    done_jobs += 1
                    elapsed = default_timer() - start
                    if elapsed >= next_report:
                        logger.info(
                            "PROGRESS: at %.2f%% sentences, %.0f sentences/s",
                            100.0 * sentence_count, sentence_count / elapsed)
                        next_report = elapsed + report_delay  # don't flood log, wait report_delay seconds
                else:
                    # loop ended by job count; really done
                    break
            except Empty:
                pass  # already out of loop; continue to next push

        elapsed = default_timer() - start
        self.clear_sims()
        logger.info(
            "scoring %i sentences took %.1fs, %.0f sentences/s",
            sentence_count, elapsed, sentence_count / elapsed)
        return sentence_scores[:sentence_count]

    def clear_sims(self):
        self.syn0norm = None

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.syn0 = np.empty((len(self.vocab), self.vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            # construct deterministic seed from word AND seed argument
            self.syn0[i] = self.seeded_vector(self.index2word[i] + str(self.seed))
        # if self.hs:
        #     self.syn1 = np.zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = np.zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        self.syn0norm = None

        self.syn0_lockf = np.ones(len(self.vocab), dtype=REAL)  # np.zeros suppress learning

    def seeded_vector(self, seed_string):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = np.random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        if self.weight_init == "uniform":
            return (once.rand(self.vector_size) - 0.5) / self.vector_size
        elif self.weight_init == "gaussian":
            return once.normal(size=self.vector_size) / self.vector_size

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.syn0.shape[0]):
                    self.syn0[i, :] /= np.sqrt((self.syn0[i, :] ** 2).sum(-1))
                self.syn0norm = self.syn0
                if hasattr(self, 'syn1'):
                    del self.syn1
            else:
                self.syn0norm = (self.syn0 / np.sqrt((self.syn0 ** 2).sum(-1))[..., np.newaxis]).astype(REAL)

    def __getitem__(self, words):

        """
        Accept a single word or a list of words as input.

        If a single word: returns the word's representations in vector space, as
        a 1D numpy array.

        Multiple words: return the words' representations in vector space, as a
        2d numpy array: #words x #vector_size. Matrix rows are in the same order
        as in input.

        Example::

          >>> trained_model['office']
          array([ -1.40128313e-02, ...])

          >>> trained_model[['office', 'products']]
          array([ -1.40128313e-02, ...]
                [ -1.70425311e-03, ...]
                 ...)

        """
        if isinstance(words, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.syn0[self.vocab[words].index]

        return np.vstack([self.syn0[self.vocab[word].index] for word in words])

    def __contains__(self, word):
        return word in self.vocab

    def similarity(self, w1, w2, unit=True):
        """
        Compute cosine similarity between two words.
    
        Example::
    
          >>> trained_model.similarity('woman', 'man')
          0.73723527
    
          >>> trained_model.similarity('woman', 'woman')
          1.0
    
        """
        if unit:
            return np.dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))
        else:
            return np.dot(self[w1], self[w2])

    def n_similarity(self, ws1, ws2, unit=True):
        """
        Compute cosine similarity between two sets of words.

        Example::

          >>> trained_model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
          0.61540466561049689

          >>> trained_model.n_similarity(['restaurant', 'japanese'], ['japanese', 'restaurant'])
          1.0000000000000004

          >>> trained_model.n_similarity(['sushi'], ['restaurant']) == trained_model.similarity('sushi', 'restaurant')
          True

        """
        v1 = [self[word] for word in ws1]
        v2 = [self[word] for word in ws2]
        if unit:
            return np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))
        else:
            return np.dot(np.array(v1).mean(axis=0), np.array(v2).mean(axis=0))


    def most_similar(self, positive=[], negative=[], topn=10, restrict_vocab=None, indexer=None):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words and the vectors for each word in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.

        If topn is False, most_similar returns the vector of similarity scores.

        `restrict_vocab` is an optional integer which limits the range of vectors which
        are searched for most-similar values. For example, restrict_vocab=10000 would
        only check the first 10000 word vectors in the vocabulary order. (This may be
        meaningful if you've sorted the vocabulary by descending frequency.)

        Example::

          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]

        """
        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (np.ndarray,)) else word
            for word in positive
            ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (np.ndarray,)) else word
            for word in negative
            ]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, np.ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.syn0norm[self.vocab[word].index])
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        limited = self.syn0norm if restrict_vocab is None else self.syn0norm[:restrict_vocab]
        dists = np.dot(limited, mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    # def wmdistance(self, document1, document2):
    #     """
    #     Compute the Word Mover's Distance between two documents. When using this
    #     code, please consider citing the following papers:
    #
    #     .. Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching".
    #     .. Ofir Pele and Michael Werman, "Fast and robust earth mover's distances".
    #     .. Matt Kusner et al. "From Word Embeddings To Document Distances".
    #
    #     Note that if one of the documents have no words that exist in the
    #     Word2Vec vocab, `float('inf')` (i.e. infinity) will be returned.
    #
    #     This method only works if `pyemd` is installed (can be installed via pip, but requires a C compiler).
    #
    #     Example:
    #         >>> # Train word2vec model.
    #         >>> model = Word2Vec(sentences)
    #
    #         >>> # Some sentences to test.
    #         >>> sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
    #         >>> sentence_president = 'The president greets the press in Chicago'.lower().split()
    #
    #         >>> # Remove their stopwords.
    #         >>> from nltk.corpus import stopwords
    #         >>> stopwords = nltk.corpus.stopwords.words('english')
    #         >>> sentence_obama = [w for w in sentence_obama if w not in stopwords]
    #         >>> sentence_president = [w for w in sentence_president if w not in stopwords]
    #
    #         >>> # Compute WMD.
    #         >>> distance = model.wmdistance(sentence_obama, sentence_president)
    #     """
    #
    #     if not PYEMD_EXT:
    #         raise ImportError("Please install pyemd Python package to compute WMD.")
    #
    #     # Remove out-of-vocabulary words.
    #     len_pre_oov1 = len(document1)
    #     len_pre_oov2 = len(document2)
    #     document1 = [token for token in document1 if token in self]
    #     document2 = [token for token in document2 if token in self]
    #     diff1 = len_pre_oov1 - len(document1)
    #     diff2 = len_pre_oov2 - len(document2)
    #     if diff1 > 0 or diff2 > 0:
    #         logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).',
    #                     diff1, diff2)
    #
    #     if len(document1) == 0 or len(document2) == 0:
    #         logger.info('At least one of the documents had no words that were'
    #                     'in the vocabulary. Aborting (returning inf).')
    #         return float('inf')
    #
    #     dictionary = Dictionary(documents=[document1, document2])
    #     vocab_len = len(dictionary)
    #
    #     # Sets for faster look-up.
    #     docset1 = set(document1)
    #     docset2 = set(document2)
    #
    #     # Compute distance matrix.
    #     distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    #     for i, t1 in dictionary.items():
    #         for j, t2 in dictionary.items():
    #             if not t1 in docset1 or not t2 in docset2:
    #                 continue
    #             # Compute Euclidean distance between word vectors.
    #             distance_matrix[i, j] = np.sqrt(np.sum((self[t1] - self[t2]) ** 2))
    #
    #     if np.sum(distance_matrix) == 0.0:
    #         # `emd` gets stuck if the distance matrix contains only np.zeros.
    #         logger.info('The distance matrix is all np.zeros. Aborting (returning inf).')
    #         return float('inf')
    #
    #     def nbow(document):
    #         d = np.zeros(vocab_len, dtype=np.double)
    #         nbow = dictionary.doc2bow(document)  # Word frequencies.
    #         doc_len = len(document)
    #         for idx, freq in nbow:
    #             d[idx] = freq / float(doc_len)  # Normalized word frequencies.
    #         return d
    #
    #     # Compute nBOW representation of documents.
    #     d1 = nbow(document1)
    #     d2 = nbow(document2)
    #
    #     # Compute WMD.
    #     return emd(d1, d2, distance_matrix)

    def most_similar_cosmul(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words, using the multiplicative combination objective
        proposed by Omer Levy and Yoav Goldberg in [4]_. Positive words still contribute
        positively towards the similarity, negative words negatively, but with less
        susceptibility to one large distance dominating the calculation.

        In the common analogy-solving case, of two positive and one negative examples,
        this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.

        Additional positive or negative examples contribute to the numerator or denominator,
        respectively - a potentially sensible but untested extension of the method. (With
        a single positive example, rankings will be the same as in the default most_similar.)

        Example::

          >>> trained_model.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'])
          [(u'iraq', 0.8488819003105164), ...]

        .. [4] Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations, 2014.

        """
        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar_cosmul('dog'), as a shorthand for most_similar_cosmul(['dog'])
            positive = [positive]

        all_words = set()

        def word_vec(word):
            if isinstance(word, np.ndarray):
                return word
            elif word in self.vocab:
                all_words.add(self.vocab[word].index)
                return self.syn0norm[self.vocab[word].index]
            else:
                raise KeyError("word '%s' not in vocabulary" % word)

        positive = [word_vec(word) for word in positive]
        negative = [word_vec(word) for word in negative]
        if not positive:
            raise ValueError("cannot compute similarity with no input")

        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        pos_dists = [((1 + np.dot(self.syn0norm, term)) / 2) for term in positive]
        neg_dists = [((1 + np.dot(self.syn0norm, term)) / 2) for term in negative]
        dists = np.prod(pos_dists, axis=0) / (np.prod(neg_dists, axis=0) + 0.000001)

        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def most_similar_cosmul_not(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words, using the multiplicative combination objective
        proposed by Omer Levy and Yoav Goldberg in [4]_. Positive words still contribute
        positively towards the similarity, negative words negatively, but with less
        susceptibility to one large distance dominating the calculation.

        In the common analogy-solving case, of two positive and one negative examples,
        this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.

        Additional positive or negative examples contribute to the numerator or denominator,
        respectively - a potentially sensible but untested extension of the method. (With
        a single positive example, rankings will be the same as in the default most_similar.)

        Example::

          >>> trained_model.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'])
          [(u'iraq', 0.8488819003105164), ...]

        .. [4] Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations, 2014.

        """
        # self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar_cosmul('dog'), as a shorthand for most_similar_cosmul(['dog'])
            positive = [positive]

        all_words = set()

        def word_vec(word):
            if isinstance(word, np.ndarray):
                return word
            elif word in self.vocab:
                all_words.add(self.vocab[word].index)
                return self.syn0[self.vocab[word].index]
            else:
                raise KeyError("word '%s' not in vocabulary" % word)

        positive = [word_vec(word) for word in positive]
        negative = [word_vec(word) for word in negative]
        if not positive:
            raise ValueError("cannot compute similarity with no input")

        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        pos_dists = [((1 + np.dot(self.syn0, term)) / 2) for term in positive]
        neg_dists = [((1 + np.dot(self.syn0, term)) / 2) for term in negative]
        dists = np.prod(pos_dists, axis=0) / (np.prod(neg_dists, axis=0) + 0.000001)

        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        """
        Find the top-N most similar words.

        If topn is False, similar_by_word returns the vector of similarity scores.

        `restrict_vocab` is an optional integer which limits the range of vectors which
        are searched for most-similar values. For example, restrict_vocab=10000 would
        only check the first 10000 word vectors in the vocabulary order. (This may be
        meaningful if you've sorted the vocabulary by descending frequency.)

        Example::

          >>> trained_model.similar_by_word('graph')
          [('user', 0.9999163150787354), ...]

        """

        return self.most_similar(positive=[word], topn=topn, restrict_vocab=restrict_vocab)

    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        """
        Find the top-N most similar words by vector.

        If topn is False, similar_by_vector returns the vector of similarity scores.

        `restrict_vocab` is an optional integer which limits the range of vectors which
        are searched for most-similar values. For example, restrict_vocab=10000 would
        only check the first 10000 word vectors in the vocabulary order. (This may be
        meaningful if you've sorted the vocabulary by descending frequency.)

        Example::

          >>> trained_model.similar_by_vector([1,2])
          [('survey', 0.9942699074745178), ...]

        """

        return self.most_similar(positive=[vector], topn=topn, restrict_vocab=restrict_vocab)

    def doesnt_match(self, words):
        """
        Which word from the given list doesn't go with the others?

        Example::

          >>> trained_model.doesnt_match("breakfast cereal dinner lunch".split())
          'cereal'

        """
        self.init_sims()

        words = [word for word in words if word in self.vocab]  # filter out OOV words
        logger.debug("using words %s" % words)
        if not words:
            raise ValueError("cannot select a word from an empty list")
        vectors = np.vstack(self.syn0norm[self.vocab[word].index] for word in words).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = np.dot(vectors, mean)
        return sorted(zip(dists, words))[0][1]

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors, recalculable table
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'table', 'cum_table'])
        super(mWord2Vec, self).save(*args, **kwargs)

    save.__doc__ = utils.SaveLoad.save.__doc__

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(mWord2Vec, cls).load(*args, **kwargs)
        # update older models
        if hasattr(model, 'table'):
            delattr(model, 'table')  # discard in favor of cum_table
        if model.negative and hasattr(model, 'index2word'):
            model.make_cum_table()  # rebuild cum_table from vocabulary
        if not hasattr(model, 'corpus_count'):
            model.corpus_count = None
        for v in model.vocab.values():
            if hasattr(v, 'sample_int'):
                break  # already 0.12.0+ style int probabilities
            elif hasattr(v, 'sample_probability'):
                v.sample_int = int(round(v.sample_probability * 2 ** 32))
                del v.sample_probability
        if not hasattr(model, 'syn0_lockf') and hasattr(model, 'syn0'):
            model.syn0_lockf = np.ones(len(model.syn0), dtype=REAL)
        if not hasattr(model, 'random'):
            model.random = np.random.RandomState(model.seed)
        if not hasattr(model, 'train_count'):
            model.train_count = 0
            model.total_train_time = 0
        return model