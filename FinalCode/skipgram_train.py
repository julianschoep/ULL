import gensim
import logging
from dgm4nlp.tf.monolingual import prepare_training
from collections import defaultdict
import operator
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import time
import _pickle as pickle
import os
import numpy as np
import random

def t():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

if 'skipgram_sents.pickle' not in os.listdir():
    data_f = open('europarl/training.en','r')
    lines = data_f.readlines()
    data_f.close()
    sentences = [x.split() for x in lines]

    stop_words = [x.strip() for x in open('stopwords.txt').readlines()]

    print(t(),'Done reading data')

    d = defaultdict(lambda:0)
    words = []
    for line in sentences:
        for word in line:
            d[word] += 1
            words.append(word)
    print(t(),'Done counting words')

    n = 30000
    sorted_x = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    vocab = [k for (k,v) in sorted_x[:n]]


    def get_wordprobs(words, counts,threshold=0.0005):
        # Source: Mikolov et al., "Distributed representations of words
        #  and phrases and their compositionality" (2013)
        # We remove words from a dataset with a probability related 
        # to the frequency of the word occurance. Frequent words 
        # are deleted more often than non-frequent words. This is done
        # to reduce the number of training samples needed.
        # Probability for removal is computed as
        #
        # P(w_i) = 1 - sqrt(t/(f(w_i)))
        #
        # Where f(w_i) is the relative frequency of word w_i and t a 
        # chosen treshold (typically 0.00001). This does not work for 
        # a small dataset. So if
        # Returns subsampled sentence set
        n = sum([counts[x] for x in list(counts.keys())])
        freqs = {word: counts[word]/n for word in list(counts.keys())}
        
        word_probs = {word: 1-np.sqrt(threshold/freqs[word]) for word in freqs.keys()}
        # word_probs['the'] returns the probability to drop 'the'
        return word_probs
        
    def get_nested(inp):
        result = []
        for a in inp:
            for j in a:
                result.append(j)
        return result

    def subsample_line(words,word_probs):
        new_words = [word for word in words if random.random() < 1-(word_probs[word])]
        
        return new_words
    


    def preprocess(lines, probs):
        new_sents = []
        n = len(lines)
        for i,line in enumerate(lines):
            new_line = []
            og_line = line
            line = subsample_line(line,probs)
            for word in line:

                if word not in vocab or word in stop_words:
                    new_line.append('<UNK>')
                else:
                    new_line.append(word.lower())
            if i % 100000 == 0:
                print(t(),'{}/{}'.format(i,n))
                print(t(),len(og_line),og_line[:6])
                print(t(),len(new_line),new_line[:6])
                print(t(),len(line),line[:6])
            new_sents.append(new_line)
        return new_sents
    s = time.time()
    word_probs = get_wordprobs(words, d)

    sentences = preprocess(sentences, word_probs)
    print(t(),'Preprocessing took {} s'.format(s-time.time()))

    with open('skipgram_sents.pickle','wb') as f:
        pickle.dump(sentences, f)

        print(t(),'Saved {} sentences'.format(len(sentences)))
else:
    with open('skipgram_sents.pickle','rb') as f:
        sentences = pickle.load(f)
if 'model.pickle' not in os.listdir():
    model = gensim.models.Word2Vec(sentences, sg=1, workers=4, min_count=1, size=100)
    with open('model.pickle','wb') as f:
        pickle.dump(model,f)
else:
    with open('model.pickle','rb') as f:
        model = pickle.load(f)

print(t(),'Writing learned vectors to file')
vec_file = open('skipgram.vec','w')
words = model.wv.vocab.keys()

def vec2str(a):
    raw = str(a).split('[')[1].split(']')[0]
    result = ''
    for el in raw.split():
        result += el.strip() + ' '
    return result.strip()


for word in words:
    vec = model.wv[word]
    line = '{} {}\n'.format(word, vec2str(vec))
    #print(line)
    vec_file.write(line)
    
vec_file.close()
print(t(),'Word vector file saved!')

