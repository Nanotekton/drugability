# make sure  numpy, scipy, pandas, sklearn are installed, otherwise run
# pip install numpy scipy pandas scikit-learn
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--labels', type=str, help= 'array of class labels. NAN or -1 represents lack of data')
parser.add_argument('--vectors', type=str, help='array of vectors representing molecules')
parser.add_argument('--names', type=str, help='Original CSV file, column names are taken from it.' )
parser.add_argument('--save', type=str, default=None)
parser.add_argument('--brf', action='store_true', help='toggles Balanced Random Forest on')

args=parser.parse_args()

import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
import gzip
import pickle

def gzpickle(name, obj):
   with gzip.open(name, 'wb') as f:
      pickle.dump(obj, f)

def gzload(name):
   with gzip.open(name, 'rb') as f:
      data = np.load(f)
      if data.dtype.__str__()=='object':
         data=data.astype(float)
      return data

# load data

with open(args.names, 'r') as f:
   names = f.readline().split(',')

labels = gzload(args.labels)
Nlabels = labels.shape[1]

assert Nlabels<len(names),'Assuming first columns are labels followed by some other data'

names = names[:Nlabels]
vectors = gzload(args.vectors)

bacc_av, auc_av = 0,0
# Build a random forest model for all twelve assays
for i, name in enumerate(names):
   column = labels[:,i]
   if -1 in column:
      finite_idx = np.where(column>=0)[0]
   else:
      finite_idx = np.where(np.isfinite(column))[0]
   x = vectors[finite_idx,:]
   y = column[finite_idx]
   if y.sum()==0 or y.sum()==len(y):
      print("%15s: undefined" % (name))
      continue
   train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, stratify=y)

   if args.brf:
    rf = BalancedRandomForestClassifier(n_estimators=100,  n_jobs=4)
   else:  
    rf = RandomForestClassifier(n_estimators=100,  n_jobs=4)
   
   rf.fit(train_x, train_y)
   p_te = rf.predict_proba(test_x)
   auc_te = roc_auc_score(test_y, p_te[:, 1])
   bacc = balanced_accuracy_score(test_y, p_te[:, 1].round(0))
   print("%15s: %3.5f %3.5f" % (name, auc_te, bacc))
   bacc_av+=bacc
   auc_av+=auc_te

   if not (args.save is None):
      gzpickle(args.save+'_%i.pkz'%i, rf)


print('Averages:')
print('AUC: %8.3f   BAcc: %8.3f'%(auc_av/(i+1), bacc_av/(i+1)))
