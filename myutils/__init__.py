import gzip
import pickle 
from numpy import load, save

def save_gz(filename, arr):
   with gzip.open(filename, 'wb') as f:
      save(f, arr)


def load_gz(filename):
   with gzip.open(filename, 'rb') as f:
      return load(f)
   

def unpickle_gz(filename):
   with gzip.open(filename, 'rb') as f:
      return pickle.load(f)


def pickle_gz(filename, obj):
   with gzip.open(filename, 'wb') as f:
      pickle.dump(obj, f, protocol=2)
    

