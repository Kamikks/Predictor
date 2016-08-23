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

reload(sys)
sys.setdefaultencoding("utf-8")

num_words = 64 # all words in train_dataset


## read and parse train data
vocab_size = 100000
raw_words = []

m = MeCab.Tagger()
with open('train.txt') as f:
  for line in f:
    for l in m.parse(line).splitlines():
      w = l.split()[0]
      raw_words.append(w)

## picked up most common vocab_size(=50000) words
index_of = {}
word_of = {}
count = [['UNK', -1]]

count.extend(collections.Counter(raw_words).most_common(vocab_size - 1))
index_of['UNK'] = 0
word_of[0] = 'UNK'
for word, _ in count:
  i = len(index_of)
  index_of[word] = i 
  word_of[i] = word

##  trans word to sampled index 
def word_to_array(word):
  data = np.zeros(shape=(vocab_size), dtype=np.int32)

  if index_of[word] > 0:
    # known word
    data[index_of[word]] = 1
  else:
    # unknown word
    data[0] = 1

  return data

## print 
print(raw_words[1])
print(index_of[raw_words[1]])
print(word_of[10])
print(word_to_array(word_of[10]))

## word_from_prob
def word_from_prob(prob):
  for i in np.argmax(prob, 1):
    if i < len(word_of):
      return [word_of[i]]
    else:
      # unknown word
      return [word_of[0]]
  #return [word_of[i] for i in np.argmax(prob, 1)]

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
num_unrollings = 30
batch_size = 1
embed_size = 200

graph = tf.Graph()
with graph.as_default():
  train_dataset = list()
  for i in range(num_unrollings): 
    train_dataset.append(tf.placeholder(tf.int32, shape=[batch_size]))

  train_labels = list()
  for i in range(num_unrollings):
    train_labels.append(tf.placeholder(tf.float32, shape=[1, vocab_size]))

  # variable
  embeddings = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1.0, 1.0))
  weight = tf.Variable(tf.truncated_normal([embed_size, vocab_size], -0.1, 0.1))
  bias = tf.Variable(tf.zeros([1, vocab_size]))

  # model
  lstm = tf.nn.rnn_cell.BasicLSTMCell(embed_size)
  state = tf.zeros([batch_size, lstm.state_size])
  saved_state = tf.Variable(tf.zeros([batch_size, lstm.state_size]), trainable=False)
  
  # unrolled lstm loop
  loss = 0.0
  outputs = list()
  with tf.variable_scope("rnn") as scope:
    for current_word in train_dataset:
      embed = tf.nn.embedding_lookup(embeddings, current_word)
      if len(outputs) > 0:
        scope.reuse_variables()
      output, state = lstm(embed, state)
      outputs.append(output) 

  with tf.control_dependencies([saved_state.assign(state)]):
    logits = tf.matmul(tf.concat(0, outputs), weight) + bias
    print(logits)
    print(tf.concat(0, train_labels))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

  ## optimizer
  optimizer = tf.train.GradientDescentOptimizer(0.3).minimize(loss)


  ## prediction
  test_input = tf.placeholder(tf.int32, shape=[1])
  test_embed = tf.nn.embedding_lookup(embeddings, test_input)
  saved_test_state = tf.Variable(tf.zeros([1, lstm.state_size]))
  test_output, test_state = lstm(test_embed, saved_state)
  with tf.control_dependencies([saved_test_state.assign(test_state)]):
    test_prediction = tf.nn.softmax(tf.matmul(test_output, weight) + bias) 



## training
num_steps = 16 
with tf.Session(graph=graph) as session:
  # initialize
  tf.initialize_all_variables().run()
  saver = tf.train.Saver()
  feed_dict = dict()

  # restore model
  if path.exists("model.saved"):
    saver.restore(session, "model.saved")

  # train
  for step in range(num_steps):
    # pick up 64 sample
    batch_index = random.sample(range(0, len(raw_words) - num_unrollings), 64)
    for cursor in batch_index:
      # train_sentence = ""
      for i in range(num_unrollings):
        feed_dict[train_dataset[i]] = [index_of[raw_words[cursor + i]]]
        feed_dict[train_labels[i]] = [word_to_array(raw_words[cursor + i + 1])]
        # train_sentence += word_from_prob([raw_words[cursor + i]])[0]
        # train_sentence += " "
      # print(train_sentence)
      _, l = session.run([optimizer, loss], feed_dict=feed_dict)

    # print loss
    if(step % 2 == 0):
      print('Loss at step %d: %f' % (step, l))

    # sample prediction
    if(step % 6 == 0):
      print("Sample Prediction:")
      for _ in range(10):
        test_data = [random.choice(index_of.values())]
        sentence = word_of[test_data[0]]
        for _ in range(20):
          prediction = test_prediction.eval({test_input: test_data})
          w = word_from_prob(sample(prediction))[0]
          #w = word_from_prob(prediction)[0]
          sentence += w
          sentence += " "
          test_data = [index_of[w]]
        print(sentence)

    # save trained model
    #shutil.copyfile("model.saved", "model.saved.bak")
    saver.save(session, "model.saved")

