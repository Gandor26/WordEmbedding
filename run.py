#from w2v import word2vec as Model
#from glove import GloVe as Model
import urllib.request
import zipfile
import argparse
import importlib
import pdb
import os

def corpus_download(filename, expected_bytes=31344016):
    path_dir = './'
    url = 'http://mattmahoney.net/dc/'
    if not os.path.exists(path_dir+filename):
        filename, _ = urllib.request.urlretrieve(url+filename, filename)
    stat = os.stat(path_dir+filename)
    if stat.st_size == expected_bytes:
        print("Found and verified corpus file '%s'" % filename)
        with zipfile.ZipFile(filename) as zf:
            if not os.path.exists(zf.namelist()[0]):
                corpus_file = zf.extract(zf.namelist()[0])
            else:
                filename = zf.namelist()[0]
    else:
        raise ValueError('Failed to verify the wanted corpus file.')
    return filename

def run(corpus_file, vocab_file, output_file):
    model = Model(corpus_file, vocab_file)
    model.train(epoch=EPOCH)
    model.output(output_file)

def main():
    #corpus_file = corpus_download('text8.zip')
    corpus_file = './test'
    vocab_file = './vocab.txt'
    output_file = './vector.txt'
    run(corpus_file, vocab_file, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a word embedding model')
    parser.add_argument('-m', '--model', type=str, nargs='?', default='glove', choices=['w2v', 'glove'], help='select a model, w2v or glove')
    parser.add_argument('-e', '--epoch', type=int, nargs='?', default=50, help='train epochs')
    parser.add_argument('-d', '--dim', type=int, nargs='?', default=100, help='vector dimension')
    parser.add_argument('-lr', '--learning_rate', type=float, nargs='?', default=1e-3, help='learning rate')
    args = parser.parse_args()

    VECTOR_DIM = args.dim
    LEARNING_RATE = args.learning_rate
    EPOCH = args.epoch
    ALPHA = .75
    THRESHOLD = 1e2

    Model = importlib.import_module(args.model).model
    main()
