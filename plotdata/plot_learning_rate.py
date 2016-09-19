import pickle
import os
import os.path as path

resultpath = path.join(path.join(os.getcwd(), '../data'), 'result.pickle')

loss_result = {}
if path.exists(resultpath):
  with open(resultpath, mode='rb') as f:
    loss_result = pickle.load(f)
else:
  print('result.pickle does not exist')

print(loss_result)



