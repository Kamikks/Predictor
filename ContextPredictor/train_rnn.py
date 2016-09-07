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
n_epochs = 501 
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, dest='epoch', nargs=1)
args = parser.parse_args()
n_epochs = args.epoch[0]


### load traindata  ###
if path.exists('traindata.pickle'):
  with open('traindata.pickle', mode='rb') as f:
    [text_raw, text_index] = pickle.load(f)
else:
  sys.exit("traindata.pickle does not exist")

### load dictionary ###
if path.exists('dictionary.pickle'):
  with open('dictionary.pickle', mode='rb') as f:
    [index_of, word_of, vocab_size] = pickle.load(f)
else:
  sys.exit("dictionary.pickle does not exist")


###  trans word to sampled index  ###
def word_to_array(word):
  data = np.zeros(shape=(vocab_size), dtype=np.float32)
  if index_of[word] > 0:
    # known word
    data[index_of[word]] = 1
  else:
    # unknown word
    data[0] = 1
  return data

def index_to_array(i):
  data = np.zeros(shape=(vocab_size), dtype=np.float32)
  data[i] = 1
  return data

## word_from_probability ###
def word_from_prob(prob):
  p = np.zeros(shape=[1, vocab_size], dtype=np.float)
  #i = random.randint(0, len(prob[0]) / 10)
  p[0, random.choice(np.argsort(prob[0])[-5:])] = 1.0
  for i in np.argmax(p, 1):
    if i < len(word_of):
      return [word_of[i]]
    else:
      # unknown word
      return [word_of[0]]

### rnn model ###
num_unrollings = 30
batch_size = 128 
embed_size = 100 
num_nodes = 64

graph = tf.Graph()
with graph.as_default():
  train_dataset = list()
  for i in range(num_unrollings): 
    train_dataset.append(tf.placeholder(tf.int32, shape=[batch_size]))

  train_labels = list()
  for i in range(num_unrollings):
    train_labels.append(tf.placeholder(tf.float32, shape=[batch_size, vocab_size]))

  # variable
  embeddings = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1.0, 1.0))
  weight = tf.Variable(tf.truncated_normal([num_nodes, vocab_size], -0.1, 0.1))
  bias = tf.Variable(tf.zeros([vocab_size]))

  # model
  lstm = tf.nn.rnn_cell.BasicLSTMCell(num_nodes)
  saved_state = tf.Variable(tf.zeros([batch_size, lstm.state_size]), trainable=False)
  
  # unrolled lstm loop
  loss = 0.0
  outputs = list()
  with tf.variable_scope("rnn") as scope:
    state = saved_state
    for current_word in train_dataset:
      embed = tf.nn.embedding_lookup(embeddings, current_word)
      if len(outputs) > 0:
        scope.reuse_variables()
      output, state = lstm(embed, state)
      outputs.append(output) 

  with tf.control_dependencies([saved_state.assign(state)]):
    logits = tf.matmul(tf.concat(0, outputs), weight) + bias
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

  ## optimizer
  optimizer = tf.train.GradientDescentOptimizer(0.7).minimize(loss)

  ## prediction
  sample_input = tf.placeholder(tf.int32, shape=[1])
  with tf.variable_scope("rnn") as scope:
    scope.reuse_variables()
    sample_embed = tf.nn.embedding_lookup(embeddings, sample_input)
    sample_saved_state = tf.Variable(tf.zeros([1, lstm.state_size]))
    sample_output, sample_state = lstm(sample_embed, sample_saved_state)
  with tf.control_dependencies([sample_saved_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.matmul(sample_output, weight) + bias) 

### training ###
skip = len(text_index) / batch_size
with tf.Session(graph=graph) as session:
  # initialize
  tf.initialize_all_variables().run()
  saver = tf.train.Saver()
  feed_dict = dict()

  # restore model
  if path.exists("model.saved"):
    saver.restore(session, "model.saved")

  # train
  for epoch in range(n_epochs):
    for i in range(num_unrollings):
      dataset = []
      labels = []

      for j in range(batch_size): 
        dataset.append(text_index[(j * skip + i) % len(text_index)])
        labels.append(index_to_array(text_index[(j * skip + i + 1) % len(text_index)]))
      
      feed_dict[train_dataset[i]] = dataset
      feed_dict[train_labels[i]] = labels

    _, l = session.run([optimizer, loss], feed_dict=feed_dict)

    # print loss
    if(epoch % 100 == 0):
      print('Loss at epoch %d: %f' % (epoch, l))

    # sample prediction
    if(epoch % 500 == 0):
      print("Sample Prediction:")
      sample_data = [random.choice(index_of.values())]
      sentence = word_of[sample_data[0]]
      for _ in range(100):
        prediction = sample_prediction.eval({sample_input: sample_data})
        w = word_from_prob(prediction)[0]
        if(w == 'EOS'):
          sentence += '\n'
        else: 
          sentence += w
        sample_data = [index_of[w]]
      print(sentence)

    # save trained model by epoch
  saver.save(session, "model.saved")

