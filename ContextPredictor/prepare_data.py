# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import collections
import MeCab
import sys
import random
import os.path as path
import shutil
import pickle
import argparse

reload(sys)
sys.setdefaultencoding("utf-8")

### argument parser ###
parser = argparse.ArgumentParser()
parser.add_argument('--trainfile', dest='trainfile', nargs=1)
args = parser.parse_args()
trainfile = args.trainfile[0]

if not path.exists(trainfile):
  sys.exit("trainfile does not exist.")


### read and parse train data ###
text_raw = []
text_index = []

# load data
if path.exists('traindata.pickle'):
  with open('traindata.pickle', mode='rb') as f:
    [text_raw, text_index] = pickle.load(f)

# create raw text data 
m = MeCab.Tagger()
with open(trainfile) as f:
  for line in f:
    for l in m.parse(line).splitlines():
      w = l.split()[0]
      text_raw.append(w)


### create dictionary ###
index_of = {}
word_of = {}
vocab_size = 2000
count = [['NaN', -1]]
index_of['NaN'] = 0
word_of[0] = 'NaN'

# load existed dictionary
if path.exists('dictionary.pickle'):
  with open('dictionary.pickle', mode='rb') as f:
    [index_of, word_of, vocab_size] = pickle.load(f)

# add new words to dictionary
count.extend(collections.Counter(text_raw).most_common(vocab_size - 1))
for word, _ in count:
  if not index_of.has_key(word):
    i = len(index_of)
    index_of[word] = i
    word_of[i] = word

# save dictionary
with open('dictionary.pickle', mode='wb') as f:
  pickle.dump([index_of, word_of, vocab_size], f)


### save traindata ###
# convert raw text data to index text data 
for w in text_raw:
  text_index.append(index_of[w])

# save traindata
with open('traindata.pickle', mode='wb') as f:
  pickle.dump([text_raw, text_index], f)


### print statistics ###
print('dictionary size: %d' % len(index_of))
print('text size: %d' % len(text_raw))

