import  basic_rnn as rnn
import tensorflow as tf
import os.path as path
import random
import numpy as np
import argparse
import shutil

num_steps = 8 

## opt parser
parser = argparse.ArgumentParser()
parser.add_argument("--sample", action="store_true")
parser.add_argument("--train", action="store_true")
args = parser.parse_args()

## load variable
text_size = rnn.text_size
num_unrollings = rnn.num_unrollings
textdata = rnn.textdata
vocab_size = rnn.vocab_size

## build tensorflow graph
graph = rnn.build_graph()

## get collection
train_dataset = graph.get_collection('train_dataset')[0]
loss = graph.get_collection('loss')[0]
optimizer = graph.get_collection('optimizer')[0]
test_prediction = graph.get_collection('test_prediction')[0]
test_input = graph.get_collection('test_input')[0]


## training
with tf.Session(graph=graph) as session:
  # initialize
  tf.initialize_all_variables().run()
  saver = tf.train.Saver()
  feed_dict = dict()

  # restore model
  if path.exists("model.saved"):
    saver.restore(session, "model.saved")

  if args.train:
    # train
    for step in range(num_steps):
      # pick up 64 sample
      batch_index = random.sample(range(0, text_size - num_unrollings), 64)
      for cursor in batch_index:
        # train_sentence = ""
        for i in range(num_unrollings):
          feed_dict[train_dataset[i]] = [textdata[cursor + i]]
          # train_sentence += word_from_prob([textdata[cursor + i]])[0]
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
          test_data = [textdata[random.randint(0, text_size-1)]]
          sentence = rnn.word_from_prob(test_data)[0]
          for _ in range(20):
            prediction = test_prediction.eval({test_input: test_data})
            w = rnn.word_from_prob(rnn.sample(prediction))[0]
            #w = rnn.word_from_prob(prediction)[0]
            sentence += w
            sentence += " "
            test_data = np.zeros(shape=(1, vocab_size), dtype=np.float)
            test_data[0, rnn.index_of[w]] = 1.0
          print(sentence)

    # save trained model
    #shutil.copyfile("model.saved", "model.saved.bak")
    saver.save(session, "model.saved")

  if args.sample:
    # sample prediction
    print("Sample Prediction:")
    for _ in range(10):
      test_data = [textdata[random.randint(0, text_size-1)]]
      sentence = rnn.word_from_prob(test_data)[0]
      for _ in range(50):
        prediction = test_prediction.eval({test_input: test_data})
        w = rnn.word_from_prob(rnn.sample(prediction))[0]
        sentence += w
        sentence += " "
        test_data = np.zeros(shape=(1, vocab_size), dtype=np.float)
        test_data[0, rnn.index_of[w]] = 1.0
      print(sentence)
      print()

