# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import collections
import sys
import random
import os.path as path
import shutil
import pickle

reload(sys)
sys.setdefaultencoding("utf-8")

def load_data(loadpath):
  trainpath = path.join(loadpath, 'traindata.pickle')
  dictpath = path.join(loadpath, 'dictionary.pickle')

  ### load traindata  ###
  if path.exists(trainpath):
    with open(trainpath, mode='rb') as f:
      [text_raw, text_index] = pickle.load(f)
  else:
    sys.exit("traindata.pickle does not exist")
  
  ### load dictionary ###
  if path.exists(dictpath):
    with open(dictpath, mode='rb') as f:
      [index_of, word_of, vocab_size] = pickle.load(f)
  else:
    sys.exit("dictionary.pickle does not exist")

  return [text_raw, text_index, index_of, word_of, vocab_size]

###  trans word to sampled index  ###
#def word_to_array(word, index_of):
#  data = np.zeros(shape=(vocab_size), dtype=np.float32)
#  if index_of[word] > 0:
#    # known word
#    data[index_of[word]] = 1
#  else:
#    # unknown word
#    data[0] = 1
#  return data

def index_to_array(i, vocab_size):
  data = np.zeros(shape=(vocab_size), dtype=np.float32)
  data[i] = 1
  return data

## word_from_probability ###
def index_from_prob(prob, max_size):
  seed = random.uniform(0.1, 0.9)
  sum = 0.0
  for i in range(len(prob[0])):
    sum += prob[0][i] 
    if sum > seed:
      if i > max_size:
        return 0
      else:
        return i
  return 0
