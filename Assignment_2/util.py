import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import tensorflow as tf

from nltk import sent_tokenize
from collections import defaultdict
import keras
from keras.models import Sequential
from keras.layers import Dense
# from keras.optimizers import SGD
import numpy as np
from nltk.corpus import stopwords

window_sz = 5  # five words left, five words right
stopwords = set(stopwords.words('english'))
sfile_path = ''


def read_input(fn):
    with open(fn, 'r') as content_file:
        content = content_file.read()
    # print(content)
    sentences = sent_tokenize(content)
    # print(sentences)
    punctuation = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '\n']

    sentences_tokens = []
    corpus = []
    reserved = ['<null>', '<unkown>']
    for sentence in sentences:
        # s= [w for w in sentence.split() if w not in punctuation]
        s = []
        for w in sentence.split():
            if w not in punctuation:
                if w in stopwords:
                    w = reserved[1]
                s.append(w)

        sentences_tokens.append(s)
        corpus = corpus + s
    corpus = set(corpus + reserved)
    print('len corpus=', len(corpus))
    word2idx, idx2word = encode_corpus(corpus)
    return word2idx, idx2word, sentences_tokens


# print(corpus)

def encode_corpus(corpus):
    word2idx = defaultdict(list)
    idx2word = defaultdict(list)
    # stpwd_idx = 2

    for idx, word in enumerate(corpus):
        word2idx[word] = idx
        idx2word[idx] = word
    return word2idx, idx2word


# print('word 0=', idx2word[0], 'word to index=', word2idx[idx2word[0]])

def onehotencoding(idx, word2idx):
    # c: corpus
    hot_enc = list()
    hot_enc = np.zeros(len(word2idx))
    # idx = word2idx[word]
    hot_enc[idx] = 1.
    # print(idx, hot_enc[idx], hot_enc)
    return hot_enc


# onehot_encoded.append(letter)


def get_features(sentences, word2idx, window_size, emb_sz):
    # sentences: set of sentences
    # word2idx dict with the word index of the whole corpus
    # window size: size of the context
    # Return: X: concatenated context and central word deterministic embeddings,
    #           shape(central_words x window_size*2 x emb_sz*2)
    #        X_hot: context hot vectors shape (central_words*window_size*2 x vocab_size)

    X_hot = []
    X = []

    R = np.random.rand(len(word2idx), emb_sz)
    for sentence in sentences:
        # print('# sentences=',len(sentences))
        # for each central word
        for idx, w_x in enumerate(sentence):
            pairs = []
            # temp = np.zeros(window_size*2, emb_sz*2)
            temp = []
            temp_hot = []
            # print('w=', w_x, len(sentence))

            for i, w_y in enumerate(sentence[max(idx - window_size, 0): \
                    min(idx + window_size, len(sentence))]):

                if idx != i:
                    temp.append(np.hstack((R[word2idx[w_x]], R[word2idx[w_y]])))
                    temp_hot.append(onehotencoding(word2idx[w_y], word2idx))

            temp = np.array(temp)
            temp_hot = np.array(temp_hot)

            # print('temp=',temp.shape)
            # pad if the contexts is smaller than window size
            if temp.shape[0] < window_size * 2:
                # print("less=",temp.shape)
                padding = window_size * 2 - temp.shape[0]
                # print("padding=", padding)
                u = [np.hstack((R[word2idx[w_x]], R[word2idx['<null>']]))]
                u_hot = [onehotencoding(word2idx['<null>'], word2idx)]

                u_all = np.repeat(u, padding, axis=0)
                u_all_hot = np.repeat(u_hot, padding, axis=0)
                if len(temp) > 0:
                    temp = np.vstack((temp, u_all))
                    temp_hot = np.vstack((temp_hot, u_all_hot))

            if len(temp) > 0:
                X_hot.append(temp_hot)
                X.append(temp)

    X_hot = np.stack(X_hot)
    X = np.stack(X)
    return X, X_hot


batch_size = 100
latent_dim = 10
intermediate_dim = 50
epochs = 50
epsilon_std = 1.0
window_size = 5
emb_sz = 50

# tr_word2idx, tr_idx2word, sent_train = read_input('./data/dev.en')
# tst_word2idx, tst_idx2word,  sent_test = read_input('./data/test.en')
# print(tr_word2idx)
# corpus_dim = len(tr_word2idx)
# original_dim = corpus_dim
# flatten_sz = (window_size*2+1)*original_dim
# context_sz=window_size*2+1
#
# x_train  = get_features(sent_train, tr_word2idx, window_size, emb_sz)
# print('shape training set=',np.array(x_train).shape)

# x_test = get_features(sent_test, tst_word2idx, window_size, emb_sz)