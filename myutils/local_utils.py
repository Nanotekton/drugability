from sys import path
path.append('/home/wbeker/projects/lingwistyka/gcnn/keras-deep-graph-learning')
from keras_dgl.layers import MultiGraphCNN, MultiGraphAttentionCNN, GraphConvLSTM
from utils import preprocess_adj_tensor_with_identity
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda
import keras.backend as K
from moldesc import process_smiles, zero_pad, adjacency_to_distance
import numpy as np
import pickle
import gzip
import logging
from rdkit import Chem

logger = logging.getLogger(__name__)

import keras.backend as K

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def tpr(y_true, y_pred):
   true=K.argmax(y_true, axis=1)
   pred=K.argmax(y_pred, axis=1)
   num_true=K.sum(true)
   true_positive=K.sum(true*pred)
   return true_positive/(num_true)#+K.epsilon())


def tnr(y_true, y_pred):
   true=K.argmin(y_true, axis=1)
   pred=K.argmin(y_pred, axis=1)
   num_true=K.sum(true)
   true_negative=K.sum(true*pred)
   return true_negative/(num_true)#+K.epsilon())


def tpr_b(y_true, y_pred):
   pred=K.round(y_pred)
   num_true=K.sum(y_true)
   true_positive=K.sum(y_true*pred)
   return true_positive/(num_true)#+K.epsilon())


def tnr_b(y_true, y_pred):
   pred=K.ones_like(y_pred)-K.round(y_pred)
   true=K.ones_like(y_true)-y_true
   num_true=K.sum(true)
   true_negative=K.sum(true*pred)
   return true_negative/(num_true)#+K.epsilon())



def _smiles_data_processor(datalines, process_smiles_config, use_semi_colon=False):
   total_X, total_A, total_lens, total_D = [], [] , [], []
   for line in datalines:
         if use_semi_colon:
            line = line.split(';')[0]
         try:
            X, A = process_smiles(line, **process_smiles_config)
         except:
            logger.error('Error with ',line)
            raise
         D = adjacency_to_distance(A, max_dist_to_include=10)
         n,f = X.shape
         total_lens.append(n)
         total_X.append(X)
         total_A.append(A)
         total_D.append(D)
   return {'X':total_X, 'A':total_A, 'L':total_lens, 'F':f, 'D':total_D}


def unpickle_gz(filename):
   with gzip.open(filename, 'rb') as f:
      return pickle.load(f)


def pickle_gz(filename, obj):
   with gzip.open(filename, 'wb') as f:
      pickle.dump(obj, f, protocol=4)
    

def simple_distance_filters(A,D):
   """Calculate distance matrices up to fifth neighbor"""
   logger.debug("Calculating distance matrices up to fifth neighbor...")
   A=preprocess_adj_tensor_with_identity(A, SYM_NORM)
   A=A.astype(np.bool)
   second = np.where(D==2,True,False)
   third = np.where(D==3,True,False)
   fourth = np.where(D==4,True,False)
   fifth = np.where(D==5,True,False)
   graph_conv_filters = np.concatenate((A,second,third,fourth,fifth),axis=1)
   graph_conv_filters = graph_conv_filters.astype(np.float64)
   return graph_conv_filters


def first_order(A,D):
   """Use first-order approximation"""
   logger.debug('Normalizing adjacency matrix')
   deg = A.sum(axis=1)
   deg = np.where(deg>0,deg**(-0.5),1)
   deg = np.array([np.diag(x) for x in deg])
   A=np.matmul(np.matmul(deg,A),deg)
   I = np.array([np.identity(x.shape[0]) for x in A])
   graph_conv_filters = np.concatenate((I,A),axis=1)
   return graph_conv_filters

def human_readable_size(size_in_bytes):
   power = int(np.log2(size_in_bytes)/10)
   divisor = 1024**power
   unit = [' B', 'KB', 'MB', 'GB']
   return '%8.1f %s'%(size_in_bytes/divisor, unit[power])


def multi_cheb(A, D, k=2):
    """Calculate Chebyshev polynomials up to order k. Works on 3D arrays."""
    logger.debug("Calculating Chebyshev polynomials up to order {}...".format(k))
    
    logger.debug('Stage 1: calculation of normalized laplacian matrices')
    del D
    L = [np.diag(x.sum(axis=1)) - x for x in A]
    del A
    L = np.array([x/np.linalg.eigvalsh(x).max() for x in L])
    size_in_mem = L.itemsize*L.size
    logger.debug('Laplacians size in memory: %s', human_readable_size(size_in_mem))
    
    T_k = list()
    T_k.append(np.array([np.identity(L.shape[1]) for _ in range(L.shape[0])]))
    T_k.append(L)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        return 2 * np.matmul(X,T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], L))
    del L
    T_k = np.concatenate(T_k, axis=1)
    size_in_mem = T_k.itemsize*T_k.size
    logger.debug('T_k size in memory: %s', human_readable_size(size_in_mem))
    return T_k


def _downsampler(range_, cutoff, randomize=False):
   assert cutoff<=range_
   if randomize:
      result = []
      while(len(result)<cutoff):
         idx=np.random.randint(0, range_-1)
         if idx not in result:
            result.append(idx)
   else:
      result=[x for x in range(range_) if x<cutoff]
   return result


FILTERS={'cheb':multi_cheb, 'first_order':first_order, 'simple_distance':simple_distance_filters}

def load_and_make_graphs(args_object, config):
   #====== choose processor =============
   if args_object.coarse_graph:
      processor = _linguistic_fragments_processor
   else:
      processor = _smiles_data_processor
   #====== read ======
   with open(args_object.positive_mols_file, 'r') as p, open(args_object.negative_mols_file, 'r') as n:
      raw_data = {}
      raw_data['positive'] = p.readlines()
      raw_data['negative'] = n.readlines()
      Lp, Ln = len(raw_data['positive']), len(raw_data['negative'])
   
   if args_object.down_sampling!='full':
      downsample_flag=True
      if Lp<Ln:
         to_process, target_len, range_ = 'negative', Lp, Ln
      elif Ln<Lp:
         to_process, target_len, range_ = 'positive', Ln, Lp
      else:
         downsample_flag=False
      if downsample_flag:
         indices = _downsampler(range_, target_len, randomize= args_object.down_sampling=='random')
         if not (args_object.save_models is None):
            pickle_gz('undersample_idx_%s.pkz'%args_object.save_models, indices)
         raw_data[to_process] = [raw_data[to_process][x] for x in indices]
   
   #====== processing =======
   logger.info('Processing positive')
   positive = processor(raw_data['positive'], config)
   logger.info('Processing negative')
   negative = processor(raw_data['negative'], config)
   
   all_smiles = raw_data['positive']+raw_data['negative']   
   #====== joining and padding ============
   max_len = max(max(positive['L']), max(negative['L']))
   F = positive['F']
   
   X, A, D, S = [], [], [], []
   lens = positive['L']+negative['L']
   Y = [1 for _ in range(len(positive['X']))] + [0 for _ in range(len(negative['X']))]
   for V in positive, negative:
      X+= [zero_pad(x, (max_len, F)) for x in V['X']]
      A+= [zero_pad(x, (max_len, max_len)) for x in V['A']]
      D+= [zero_pad(x, (max_len, max_len)) for x in V['D']]
      if args_object.coarse_graph:
         S+=[zero_pad(x, (max_len, max_len)) for x in V['S']]
   
   N=len(Y)
   idx = [x for x in range(N)]
   np.random.shuffle(idx)
   result = {}
   result['X'] = np.array([X[i] for i in idx])
   result['A'] = np.array([A[i] for i in idx])
   result['D'] = np.array([D[i] for i in idx])
   result['Y'] = np.array([Y[i] for i in idx])
   result['L'] = np.array([lens[i] for i in idx])
   result['smiles'] = [all_smiles[i] for i in idx]
   if args_object.coarse_graph:
      result['S'] = np.array([S[i] for i in idx])
   

   if not (args_object.save is None):
      pickle_gz(args_object.save, result)
      logger.info('Data saved to %s'%args_object.save)

   return result


import types
import tempfile
import keras.models

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d
   
    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name, custom_objects={'MultiGraphCNN': MultiGraphCNN, 'MultiGraphAttentionCNN':MultiGraphAttentionCNN,'tpr':tpr,'tnr':tnr, 'tnr_b':tnr_b,'tpr_b':tpr_b})
        self.__dict__ = model.__dict__
   

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

def transform_adjacency_matrix(A,S, keep_diagonal=True):
   L=np.diag(A.sum(axis=0))-A
   L_transformed = S.T.dot(L).dot(S)   
   if keep_diagonal:
      A_transformed = np.where(L!=0,1,0)
   else:
      A_transformed = np.where(L<0,1,0)
   return A_transformed


def transform_adjacency_matrices(A,S, keep_diagonal=True):
   Ls=np.array([np.diag(x.sum(axis=0))-x for x in A])
   ST = S.transpose(0,2,1)
   L_transformed = np.matmul(np.matmul(ST,L),S)
   if keep_diagonal:
      A_transformed = np.where(L!=0,1,0)
   else:
      A_transformed = np.where(L<0,1,0)
   return A_transformed


