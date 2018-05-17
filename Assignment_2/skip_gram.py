# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
import tensorflow as tf
import random
import time
import pickle
#os.listdir('small_dataset')

# All methods

def preprocess(text):

    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', '')
    text = text.replace(',', '')
    text = text.replace('"', '')
    text = text.replace(';', '')
    text = text.replace('!', '')
    text = text.replace('?', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('--', '')
    text = text.replace('?', '')
    text = text.replace(':', '')
    # Remove digits from text
    text = ''.join([i for i in text if not i.isdigit()])
    words = text.split()
    
    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = []
    for word in words:
        if word_counts[word] > 5:
            trimmed_words.append(word)
        elif word_counts[word] > 2:
            trimmed_words.append('<UNK>')
        # Words occuring less than 3 times in the vocabulary are thrown out
    

    return trimmed_words

def read_in_words(filename):
    file = open(filename,'r')
    text = file.read()
    words = preprocess(text)
    return words

def get_vocabulary_counts(word_set):
    counts = defaultdict(lambda:0)
    for word in word_set:
        counts[word] += 1
    return counts

def get_subsample_probs(freqs,t):
    n = sum([freqs[x] for x in list(freqs.keys())])
    prob_dict = defaultdict(float)
    for word in V_dict.keys():
        count = V_dict[word]
        f_w = count/n # Relative frequency
        P_keep = 1 - np.sqrt(t/f_w)
        prob_dict[word] = P_keep
    return prob_dict
    
def subsample(words, counts,threshold=0.001):
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
    
   
    new_words = [word for word in words if random.random() < 1-(word_probs[word])]
    
    return new_words
    

def get_targets(words, idx, W):
    # Returns words with a random window size in range of W
    R = np.random.randint(1,W+1)
    ixlo = max(idx-R,0)
    ixhi = min(idx+R+1,len(words))
    targets = set([word for word in words[ixlo:ixhi] if word != words[idx]])
    return list(targets)



    
def one_hot(word_int,V):
    v = np.zeros(V,dtype=int)
    v[word_int] = 1
    return v

# Surpress deprication warnings

import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
#!mkdir checkpoints


# Parameters

W = 5 # Window size
H = 300 # Number of neurons in hidden/embedding layer
epochs = 10 # number of epochs
batch_size = 1000

# Building data
start = time.time()
words_text = read_in_words('large_dataset/training.en')
# Get word counts from the corpus
counts = get_vocabulary_counts(words_text)
# Subsample corpus based on word counts/frequencies.
word_set = subsample(words_text,counts)


# Get distinct words from corpus
words = set(word_set)
# Build usefull word2int and reverse dictionary.
word2int={word:i for i,word in enumerate(words)}
int2word={i:word for i,word in enumerate(words)}

with open('w2i.skip','wb') as f:
    pickle.dump(word2int, f)

with open('i2w.skip','wb') as f:
    pickle.dump(int2word,f)

print('word2int & int2word pickled successfully.')
train_words = [word2int[word] for word in word_set]
V = len(words) # Vocabulary size
print("Reading in and preprocessing data took {} seconds.".format(time.time()-start))
#data_pairs = get_skipgram_data(sentence_set, W)
#x_train, y_train = get_xy_train(data_pairs, V)

print("{} words in vocabulary.".format(V))

def get_batches(words, batch_size, W):
    n_batches = len(words) // batch_size
    print("Num batches: {}".format(n_batches))
    words = words[:n_batches*batch_size]
    for i in range(0,len(words),batch_size):
        x,y = [],[]
        batch = words[i:i+batch_size]
        for j in range(len(batch)):
            batch_x = batch[j]
            batch_y = get_targets(batch,j,W)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x,y


# Model architecture
g = tf.Graph()

with g.as_default():
    # Input words
    inputs = tf.placeholder(tf.int32,[None],name='inputs')
    # Words that appear in the context of these input words
    labels = tf.placeholder(tf.int32, [None,None],name='labels')
    # The embedding matrix that we want to obtain/ train
    embedding = tf.Variable(tf.random_uniform((V, H), -1, 1),name='embedding')
    
    # This gives the hidden layer output for each of the inputs
    embed = tf.nn.embedding_lookup(embedding, inputs)
    
    # The weights between the hidden layer output and 
    softmax_w = tf.Variable(tf.truncated_normal((V,H)))
    softmax_b = tf.Variable(tf.zeros(V),name='softmax_bias')
    loss = tf.nn.sampled_softmax_loss(
        weights = softmax_w,
        biases  = softmax_b,
        labels  = labels,
        inputs  = embed,
        num_sampled = 100,
        num_classes = V)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    saver = tf.train.Saver()

    ## From Thushan Ganegedara's implementation
    valid_size = 16 # Randompick set of words to evaluate similarity on.
    valid_window = 100
    # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, 
                               random.sample(range(1000,1000+valid_window), valid_size//2))

    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
    


print("Starting training")
with tf.Session(graph=g) as sess:
    i = 1
    loss = 0
    init = tf.global_variables_initializer()
    sess.run(init)
    print("Starting first epoch")
    for e in range(1,epochs+1):

        batches = get_batches(train_words,batch_size,W)

        start = time.time()
        
        for x,y in batches:
            # for each batch
            feed = {inputs:x,
                    labels: np.array(y)[:,None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            loss += train_loss
            if i % 100 == 0:
                print("Epoch {}/{}".format(e,epochs),
                      "Iteration {}".format(i),
                     "Avg. Trainig loss: {:.4f}".format(loss/100),
                     "{:.4f} sec/batch".format((time.time()-start)/100))
                loss = 0
                start = time.time()
            if i % 1000 == 0:
                ## From Thushan Ganegedara's implementation
                # note that this is expensive (~20% slowdown if computed every 500 steps)
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int2word[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = int2word[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
            
            i += 1
    save_path = saver.save(sess,"skip_checkpoints/text8.ckpt")
    #embed_mat = sess.run(normalized_embedding)
    
#W1 = tf.Variable(tf.random_normal([V,H])) # W input --> hidden
#b1 = tf.Variable(tf.random_normal([H]))   # bias --> hidden

#hidden_layer = tf.add(tf.matmul(x,W1),b1)

#W2 = tf.Variable(tf.random_normal([H,V])) # W hidden --> output
#b2 = tf.Variable(tf.random_normal([V]))   # bias --> output

#out = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, W2),b2))


print("WE ARE DONE TRAINING AND IT ONLY TOOK {} SECONDS :D :D:D:D".format(time.time()-start))




