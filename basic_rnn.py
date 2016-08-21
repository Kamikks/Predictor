# -*- coding:utf-8 -*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
import collections
import MeCab
import sys
import random

reload(sys)
sys.setdefaultencoding("utf-8")

batch_size = 1
num_words = 64 # all words in train_dataset

index_of = {}
word_of = {}
counts_of = collections.Counter()
text = []



## read and parse train data
m = MeCab.Tagger()
with open('train.txt') as f:
  for line in f:
    for l in m.parse(line).splitlines():
      w = l.split()[0]
      if w not in index_of:
        i = len(word_of)
        word_of[i] = w
        index_of[w] = i
      counts_of[index_of[w]] += 1
      text.append(index_of[w])
         
## print word_of
# for w in word_of.values():
#   print(w)

## initialize
vocab_size = len(counts_of)
text_size = len(text)

textdata = np.zeros(shape=(text_size, vocab_size), dtype=np.float)
for i in range(text_size):
  textdata[i, text[i]] = 1.0


## word_from_prob
def word_from_prob(prob):
  return [word_of[i] for i in np.argmax(prob, 1)]

## sampling word from prediction
def sample_distribution(distribution):
  r = random.uniform(0.2, 0.8)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  p = np.zeros(shape=[1, vocab_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

## rnn model
num_unrollings = 50
train_dataset = list()

def build_graph():
  graph = tf.Graph()
  with graph.as_default():
    train_dataset = list()
    for i in range(num_unrollings): 
      train_dataset.append(tf.placeholder(tf.float32, shape=[batch_size, vocab_size]))
    train_inputs = train_dataset[:len(train_dataset) - 1] 
    train_labels = train_dataset[1:]
    # variable
    weight = tf.Variable(tf.truncated_normal([vocab_size, vocab_size], -0.1, 0.1))
    bias = tf.Variable(tf.zeros([1, vocab_size]))

    # model
    lstm = tf.nn.rnn_cell.BasicLSTMCell(vocab_size)
    state = tf.zeros([batch_size, lstm.state_size])
    saved_state = tf.Variable(tf.zeros([batch_size, lstm.state_size]), trainable=False)
  
    loss = 0.0
    outputs = list()
    with tf.variable_scope("rnn") as scope:
      for current_word in train_inputs:
        if len(outputs) > 0:
          scope.reuse_variables()
        output, state = lstm(current_word, state)
        outputs.append(output) 

    with tf.control_dependencies([saved_state.assign(state)]):
      logits = tf.matmul(tf.concat(0, outputs), weight) + bias
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

    ## optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.3).minimize(loss)


    ## prediction
    test_input = tf.placeholder(tf.float32, shape=[1, vocab_size])
    saved_test_state = tf.Variable(tf.zeros([1, lstm.state_size]))
    test_output, test_state = lstm(test_input, saved_state)
    with tf.control_dependencies([saved_test_state.assign(test_state)]):
      test_prediction = tf.nn.softmax(tf.matmul(test_output, weight) + bias) 


    ## add to collection
    tf.add_to_collection('train_dataset', train_dataset)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('optimizer', optimizer)
    tf.add_to_collection('test_prediction', test_prediction) 
    tf.add_to_collection('test_input', test_input)
 

  return graph


