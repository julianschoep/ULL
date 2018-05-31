from __future__ import absolute_import, division, unicode_literals
print(1)
import sys
print(2)
import numpy as np
print(3)
import logging
print(4)
import sklearn
print(5)
#import data 
# data.py is part of Senteval and it is used for loading word2vec style files
import senteval
print(6)
import tensorflow as tf
print(7)
import logging
print(8)
from collections import defaultdict
print(9)
import dill
print(10)
import dgm4nlp

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class EmbeddingExtractor:
    """
    This will compute a forward pass with the inference model of EmbedAlign and 
        give you the variational mean for each L1 word in the batch.
        
    Note that this takes monolingual L1 sentences only (at this point we have a traiend EmbedAlign model
        which dispenses with L2 sentences).    
        
    You don't really want to touch anything in this class.
    """

    def __init__(self, graph_file, ckpt_path, config=None):        
        g1 = tf.Graph()
        self.meta_graph = graph_file
        self.ckpt_path = ckpt_path
        
        self.softmax_approximation = 'botev-batch' #default
        with g1.as_default():
            self.sess = tf.Session(config=config, graph=g1)
            # load architecture computational graph
            self.new_saver = tf.train.import_meta_graph(self.meta_graph)
            # restore checkpoint
            self.new_saver.restore(self.sess, self.ckpt_path) #tf.train.latest_checkpoint(
            self.graph = g1  #tf.get_default_graph()
            # retrieve input variable
            self.x = self.graph.get_tensor_by_name("X:0")
            # retrieve training switch variable (True:trianing, False:Test)
            self.training_phase = self.graph.get_tensor_by_name("training_phase:0")
            #self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")

    def get_z_embedding_batch(self, x_batch):
        """
        :param x_batch: is np array of shape [batch_size, longest_sentence] containing the unique ids of words
        
        :returns: [batch_size, longest_sentence, z_dim]        
        """
        # Retrieve embeddings from latent variable Z
        # we can sempale several n_samples, default 1
        try:
            z_mean = self.graph.get_tensor_by_name("z:0")
            
            feed_dict = {
                self.x: x_batch,
                self.training_phase: False,
                #self.keep_prob: 1.

            }
            z_rep_values = self.sess.run(z_mean, feed_dict=feed_dict) 
        except Exception as e:
            print(e)
            raise ValueError('hejoo'+str(e))
        return z_rep_values

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#



# Set PATHs
# path to senteval
#PATH_TO_SENTEVAL = '../'



# import SentEval
#sys.path.insert(0, PATH_TO_SENTEVAL)

# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
# for example the path to a dictionary of indices, of hyper parameters
# this dictionary is passed to the batched and the prepare fucntions
params_senteval = {'task_path': '',
                   'usepytorch': True,
                   'kfold': 10,
                   'ckpt_path': '',
                   'tok_path': '',
                   'extractor': None,
                   'tks1': None}
# made dictionary a dotdict
params_senteval = dotdict(params_senteval)
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# usepytorch = False 
#params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                 'tenacity': 3, 'epoch_size': 2}



def prepare(params, samples):
    """
    In this example we are going to load a tensorflow model, 
    we open a dictionary with the indices of tokens and the computation graph
    """
    params.extractor = EmbeddingExtractor(
        graph_file='%s.meta'%(params.ckpt_path),
        ckpt_path=params.ckpt_path,
        config=None #run in cpu
    )

    # load tokenizer from training
    params.tks1 = dill.load(open(params.tok_path, 'rb'))
    return

def batcher(params, batch):
    """
    At this point batch is a python list containing sentences. Each sentence is a list of tokens (each token a string).
    The code below will take care of converting this to unique ids that EmbedAlign can understand.
    
    This function should return a single vector representation per sentence in the batch.
    In this example we use the average of word embeddings (as predicted by EmbedAlign) as a sentence representation.
    
    In this method you can do mini-batching or you can process sentences 1 at a time (batches of size 1).
    We choose to do it 1 sentence at a time to avoid having to deal with masking. 
    
    This should not be too slow, and it also saves memory.
    """
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    for sent in batch:
        # Here is where dgm4nlp converts strings to unique ids respecting the vocabulary
        # of the pre-trained EmbedAlign model
        # from tokens ot ids position 0 is en
        x1 = params.tks1[0].to_sequences([(' '.join(sent))])
        
        # extract word embeddings in context for a sentence
        # [1, sentence_length, z_dim]
        z_batch1 = params.extractor.get_z_embedding_batch(x_batch=x1)
        # sentence vector is the mean of word embeddings in context
        # [1, z_dim]
        sent_vec = np.mean(z_batch1, axis=1)
        # check if there is any NaN in vector (they appear sometimes when there's padding)
        if np.isnan(sent_vec.sum()):
            sent_vec = np.nan_to_num(sent_vec)        
        embeddings.append(sent_vec)
    embeddings = np.vstack(embeddings)
    return embeddings


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # define paths
    # path to senteval data
    # note senteval adds downstream into the path
    params_senteval.task_path = './SentEval/data/' 
    # path to computation graph
    # we use best model on validation AER
    # TODO: you have to point to valid paths! Use the pre-trained model linked from the top of this notebook.
    params_senteval.ckpt_path = './ull-practical3-embedalign/model.best.validation.aer.ckpt'
    # path to tokenizer with ids of trained Europarl data
    # out dictionary id depends on dill for pickle
    params_senteval.tok_path = './ull-practical3-embedalign/tokenizer.pickle'
    # we use 10 fold cross validation
    params_senteval.kfold = 10
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    # here you define the NLP taks that your embedding model is going to be evaluated
    # in (https://arxiv.org/abs/1802.05883) we use the following :
    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression), 
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    transfer_tasks = ['SICKEntailment','STS14']#['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                     # 'MRPC', 'SICKEntailment', 'STS14']
    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    print(results)
