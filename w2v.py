from collections import Counter
from tqdm import tqdm
import numpy as np
import pickle as pk
import os

class N_Gram(object):
    def __init__(self, *tokens):
        self.tokens = tokens
        self.score = .0
    def __str__(self):
        return ' '.join(self.tokens)
    def __hash__(self):
        return hash((self.tokens, self.score))

def corpus_gen(corpus_file):
    with open(corpus_file, 'r') as fp:
        for line in fp:
            yield line

def build_vocab(corpus, target_words):
    cnter = Counter()
    for line in corpus:
        words = line.strip().split()
        cnter.update(words)
    vocab = {word: (index, cnter[word]) for index, word in enumerate(target_words)}
    return vocab, cnter

def build_samples(vocab, cnter, corpus, sample_file='./samples.pk', context_window_size=5, threshold=1e-5):
    tokens = list()
    samples = list()
    total = sum(cnter.values()) * threshold
    for line in corpus:
        words = [word.lower() for word in line.strip().split()\
                if word.isalnum() and len(word)>1]
        tokens.clear()
        for word in tqdm(words):
            keep_prob = min(1, np.sqrt(total/cnter[word]))
            keep = np.random.binomial(1, keep_prob)
            if keep == 1:
                tokens.append(vocab.get(word, (None, 0))[0])
        for idx, token in tqdm(enumerate(tokens)):
            if token is None: continue
            s = max(0, idx-context_window_size)
            t = min(idx+context_window_size, len(tokens)-1)
            context = tokens[s:idx]+tokens[idx+1:t+1]
            while None in context:
                context.remove(None)
            if not context: continue
            samples.append((token, context))
    with open(sample_file, 'wb') as fp:
        pk.dump(samples, fp)
    return samples

def build_trainset(vocab, cnter, corpus, sample_file, context_window_size=5, threshold=1e-5):
    if not os.path.exists(sample_file):
        samples = build_samples(vocab, cnter, corpus, sample_file, context_window_size, threshold)
    else:
        with open(sample_file, 'rb') as fp:
            samples = pk.load(fp)
    print('Number of samples: %d' % len(samples))
    return samples

def sigmoid(x):
    if x > 6:
        return 1.0-1e-8
    elif x < -6:
        return 1e-8
    else:
        return 1/(1+np.exp(-x))

class model(object):
    def __init__(self, corpus_file, vocab_file, vector_dim=300, learning_rate=1e-3, alpha=.75, num_nsamples=5):
        self.target_words = list()
        for line in corpus_gen(vocab_file):
            word = line.strip().split()[0].lower()
            self.target_words.append(word)
        self.vocab, cnter = build_vocab(corpus_gen(corpus_file), self.target_words)
        self.vocab_size = len(self.vocab)
        self.tokens, prob = zip(*self.vocab.values())
        prob = np.array(prob)**alpha
        self.prob = prob / prob.sum()
        self.samples = build_trainset(self.vocab, cnter, corpus_gen(corpus_file), sample_file='./samples.pk')
        del prob, cnter

        self.vector_dim = vector_dim
        self.learning_rate = learning_rate
        self.num_nsamples = num_nsamples

    def neg_sampler(self, num_samples, *pos_samples):
        samples = np.random.choice(self.tokens, num_samples, False, self.prob)
        while any([sample in pos_samples for sample in samples]):
            samples = np.random.choice(self.tokens, num_samples, False, self.prob)
        return samples

    def optimizer(self):
        nll = 0.0
        np.random.shuffle(self.samples)
        for center_token, context in tqdm(self.samples):
            neg_samples = self.neg_sampler(self.num_nsamples, center_token, *context)
            for context_token in context:
                g_w = np.zeros(self.vector_dim)
                targets = [(context_token, 1)] + [(sample, 0) for sample in neg_samples]
                for target, label in targets:
                    z = sigmoid(np.dot(self.emb_w[center_token], self.emb_c[target]))
                    g_c = (label-z)*self.emb_w[center_token]
                    g_w += (label-z)*self.emb_c[target]
                    self.emb_c[target] += self.learning_rate * g_c / np.sqrt(self.g_sq_emb_c[target])
                    self.g_sq_emb_c[target] += np.square(g_c)
                    nll -= np.log((1-label)-(-1)**label*z)
                self.emb_w[center_token] += self.learning_rate * g_w / np.sqrt(self.g_sq_emb_w[center_token])
                self.g_sq_emb_w[center_token] += np.square(g_w)
        return nll / len(self.samples)

    def train(self, epoch=50, load=False, ckpt_file=None):
        if load:
            if ckpt_file is None:
                raise ValueError('Checkpoint file must be provided if loading pretrained embeddings')
            self.load_model(ckpt_file)
        else:
            self.emb_w = np.random.uniform(-.5, .5, size=(self.vocab_size, self.vector_dim))
            self.emb_c = np.random.uniform(-.5, .5, size=(self.vocab_size, self.vector_dim))
        self.g_sq_emb_w = np.ones((self.vocab_size, self.vector_dim))
        self.g_sq_emb_c = np.ones((self.vocab_size, self.vector_dim))

        for e in range(epoch):
            loss = self.optimizer()
            print('Epoch %d nll %.4f' % (e+1, loss))
            self.embedding = self.emb_w / np.linalg.norm(self.emb_w, axis=-1).reshape(-1, 1)
            self.output('./w2v_vector_epoch_{}.txt'.format(e+1))
        self.save_model(ckpt_file='./w2v_epoch_{}.ckpt'.format(epoch))

    def output(self, output_file='./vector.txt'):
        with open(output_file, 'w') as ofp:
            for word in self.target_words:
                vector = ['%.6f'%i for i in self.embedding[self.vocab[word][0]]]
                string = word + ' ' + ' '.join(vector) + '\n'
                ofp.write(string)

    def save_model(self, ckpt_file='w2v.ckpt'):
        with open(ckpt_file, 'wb') as cfp:
            pk.dump((self.emb_w, self.emb_c), cfp)

    def load_model(self, ckpt_file):
        with open(ckpt_file, 'rb') as cfp:
            self.emb_w, self.emb_c = pk.load(cfp)


