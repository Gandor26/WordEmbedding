from collections import Counter, namedtuple, deque
from scipy import sparse
from tqdm import tqdm
import pickle as pk
import numpy as np
import os

def corpus_gen(corpus_file):
    with open(corpus_file, 'r') as f:
        for line in f:
            yield line

def build_vocab(corpus, target_words):
    cnter = Counter()
    for line in corpus:
        words = line.strip().split()
        cnter.update(words)
    vocab = {word: (index, cnter[word]) for index, word in enumerate(target_words)}
    return vocab

def build_cooc_mat(vocab, corpus, mat_file='./cooc.pk', context_window_size=15):
    vocab_size = len(vocab)
    cooc_mat = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float64)

    for line_idx, line in enumerate(corpus):
        if line_idx % 1000 == 0:
            print('Building cooccrurence matrix on line %d' % line_idx)
        words = line.strip().split()
        tokens = [vocab.get(word, (None, 0))[0] for word in words]
        extended_tokens = [None]*(context_window_size-1) + tokens
        window = deque(maxlen=context_window_size)
        window.extend(extended_tokens[:context_window_size])
        for center_token in tqdm(extended_tokens[context_window_size:]):
            if center_token is None: continue
            for context_idx, context_token in enumerate(window):
                if context_token is None: continue
                cooc_mat[center_token, context_token] += 1.0/(context_window_size-context_idx)
                cooc_mat[context_token, center_token] += 1.0/(context_window_size-context_idx)
            window.append(center_token)
    with open(mat_file, 'wb') as fp:
        pk.dump(cooc_mat, fp)
    return cooc_mat

def build_cooc(vocab, corpus, mat_file, context_window_size=15):
    cooc = list()
    if not os.path.exists(mat_file):
        cooc_mat = build_cooc_mat(vocab, corpus, mat_file)
    else:
        with open(mat_file, 'rb') as fp:
            cooc_mat = pk.load(fp)
    for center_token, (row, data_row) in enumerate(zip(cooc_mat.rows, cooc_mat.data)):
        for context_token, data in zip(row, data_row):
            cooc.append((center_token, context_token, data))
    print('Number of samples: %d' % len(cooc))
    return cooc


class model(object):
    def __init__(self, corpus_file, vocab_file, vector_dim=100, learning_rate=1e-3, alpha=.75, x_max=1e2):
        self.target_words = list()
        for line in corpus_gen(vocab_file):
            word = line.strip().split()[0].lower()
            self.target_words.append(word)
        self.vocab = build_vocab(corpus_gen(corpus_file), self.target_words)
        self.vocab_size = len(self.vocab)
        self.cooc = build_cooc(self.vocab, corpus_gen(corpus_file), mat_file='./cooc.pk')

        self.vector_dim = vector_dim
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.x_max = x_max

    def optimizer(self):
        loss = 0.0
        np.random.shuffle(self.cooc)
        for center_token, context_token, cooccurence in tqdm(self.cooc):
            e_w = self.emb_w[center_token]
            e_c = self.emb_c[context_token]
            b_w = self.bias_w[center_token]
            b_c = self.bias_c[context_token]
            g_sq_e_w = self.g_sq_emb_w[center_token]
            g_sq_e_c = self.g_sq_emb_c[context_token]
            g_sq_b_w = self.g_sq_bias_w[center_token]
            g_sq_b_c = self.g_sq_bias_c[context_token]

            weight = (cooccurence/self.x_max)**self.alpha if cooccurence < self.x_max else 1
            residue = np.dot(e_w, e_c) + b_w + b_c - np.log(cooccurence)
            loss += .5 * weight * residue**2

            g_emb_w = weight*e_c*residue
            g_emb_c = weight*e_w*residue
            g_bias_w = weight*residue
            g_bias_c = weight*residue

            e_w -= self.learning_rate*g_emb_w/np.sqrt(g_sq_e_w)
            e_c -= self.learning_rate*g_emb_c/np.sqrt(g_sq_e_c)
            b_w -= self.learning_rate*g_bias_w/np.sqrt(g_sq_b_w)
            b_c -= self.learning_rate*g_bias_c/np.sqrt(g_sq_b_c)

            g_sq_e_w += np.square(g_emb_w)
            g_sq_e_c += np.square(g_emb_c)
            g_sq_b_w += np.square(g_bias_w)
            g_sq_b_c += np.square(g_bias_c)

        return loss/len(self.cooc)

    def train(self, epoch=50, merge=True, ckpt_file=None):
        if ckpt_file is None:
            self.emb_w = np.random.uniform(-.5, .5, size=(self.vocab_size, self.vector_dim))
            self.emb_c = np.random.uniform(-.5, .5, size=(self.vocab_size, self.vector_dim))
            self.bias_w = np.random.uniform(-.5, .5, size=self.vocab_size)
            self.bias_c = np.random.uniform(-.5, .5, size=self.vocab_size)
        else:
            self.load_model(ckpt_file)
        self.g_sq_emb_w = np.ones((self.vocab_size, self.vector_dim))
        self.g_sq_emb_c = np.ones((self.vocab_size, self.vector_dim))
        self.g_sq_bias_w = np.ones(self.vocab_size)
        self.g_sq_bias_c = np.ones(self.vocab_size)

        for e in range(epoch):
            loss = self.optimizer()
            print('Epoch %d loss %.4f' % (e+1, loss))
            self.embedding = (self.emb_w + self.emb_c)/2 if merge else self.emb_w
            self.embedding /= np.linalg.norm(self.embedding, axis=-1).reshape(-1,1)
            self.output('./glove_vector_epoch_{}.txt'.format(e+1))
        self.save_model(ckpt_file='./glove_epoch_{}.ckpt'.format(epoch))

    def output(self, output_file='./vector.txt'):
        with open(output_file, 'w') as ofp:
            for word in self.target_words:
                vector = ['%.6f'%i for i in self.embedding[self.vocab[word][0]]]
                string = word + ' ' + ' '.join(vector) + '\n'
                ofp.write(string)

    def save_model(self, ckpt_file='glove.ckpt'):
        with open(ckpt_file, 'wb') as cfp:
            pk.dump((self.emb_w, self.emb_c, self.bias_w, self.bias_c), cfp)

    def load_model(self, ckpt_file):
        with open(ckpt_file, 'rb') as cfp:
            self.emb_w, self.emb_c, self.bias_w, self.bias_c = pk.load(cfp)


