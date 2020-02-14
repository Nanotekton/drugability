import numpy as np
import gzip
import pickle 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.utils.class_weight import compute_sample_weight
import yaml
from os.path import isfile 
from myutils.graph_utils import FILTERS
from myutils.moldesc import process_smiles, adjacency_to_distance, zero_pad
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem

def balance_masked_weights(y,w):
   if w is None: 
      return None
   n_pos = (y*w).sum()
   n_neg = ((1-y)*w).sum()
   n_tot = w.sum()
   if 0 in [n_pos, n_neg]:
      return w
   new_w = np.where(y==1.0, float(n_neg)/n_tot, float(n_pos)/n_tot)
   return new_w*w


def load_yaml(name):
   with open(name, 'r') as f:
      return yaml.load(f)


def gzload(name, f32=True):
   with gzip.open(name, 'rb') as f:
      data = np.load(f)
      if f32:
         data = np.nan_to_num(data.astype(np.float32))
      return data


def gzsave(name, arr):
   with gzip.open(name, 'wb') as f:
      np.save(f, arr)


def gz_pickle(name, obj):
   with gzip.open(name, 'wb') as f:
      pickle.dump(obj, f)


def gz_unpickle(name):
   with gzip.open(name, 'rb') as f:
      return pickle.load(f)


def __balance_data(data_pos, data_neg, data_balance):
   lenghts = len(data_pos), len(data_neg)
   min_shape = min(lenghts)
   max_shape = max(lenghts)
   
   if data_balance == 'cut':
      data_pos = data_pos[:min_shape,:]
      data_neg = data_neg[:min_shape,:]
   elif data_balance == 'random':
      idx = np.random.choice(np.arange(max_shape), min_shape, replace=False)
      if max_shape==lenghts[0]:
         data_pos = data_pos[idx,:]
      else:
         data_neg = data_neg[idx,:]

   print('pos,neg shapes:', data_pos.shape, data_neg.shape)
   x= np.vstack((data_pos, data_neg))
   y= np.array([1 for _ in range(data_pos.shape[0])] + [0 for _ in range(data_neg.shape[0])])

   if data_balance == 'weights':
      weights = compute_sample_weight('balanced', y)
   else:
      weights=None
   
   return x, y, weights


def numpy_load(name):
   ext = name[-3:]
   if ext=='npy':
      return np.load(name)
   elif ext=='npz':
      return gzload(name)
   else:
      raise IOError('Unknown extension %s'%ext)


def _normalize(x, idx, standarize):
   if type(idx).__name__=='str':
      idx_f=idx
      mu_f, std_f = None, None
   else:   
      mu_f, std_f, idx_f = idx
   if idx_f is None:
      return x
   if mu_f is None: mu_f = 'mean_' + idx_f
   if std_f is None: std_f = 'std_' + idx_f

   IDX = numpy_load(idx_f).astype(int)
   if standarize:
      std = numpy_load(std_f)
      mu = numpy_load(mu_f)
      x=np.nan_to_num((x-mu)/std)
   x=x[:,IDX]
   
   return x


def __load_data(data_pos, data_neg, data_checkpoint, data_balance, idx, standarize=True):
   data_pos = np.nan_to_num(gzload(data_pos))
   data_neg = np.nan_to_num(np.vstack([gzload(x) for x in data_neg]))
   x, y, weights = __balance_data(data_pos, data_neg, data_balance)
   
   x = _normalize(x, idx, standarize)

   y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1,1))
   gz_pickle(data_checkpoint, (x, y, weights))
   return x, y, weights


def load_data(data_pos, data_neg, data_checkpoint, data_balance, idx, standarize=True, force=False):
   if None in [data_pos, data_neg] or not isfile(data_checkpoint):
      force=True
   if force:
      return __load_data(data_pos, data_neg, data_checkpoint, data_balance, idx, standarize)
   else:
      return gz_unpickle(data_checkpoint)


def _smiles_data_processor(datalines, process_smiles_config, use_semi_colon=False):
   total_X, total_A, total_lens, total_D = [], [] , [], []
   for line in datalines:
         if use_semi_colon:
            line = line.split(';')[0]
         try:
            X, A = process_smiles(line, **process_smiles_config)
         except:
            print('Error with ',line)
            raise
         D = adjacency_to_distance(A, max_dist_to_include=10)
         n,f = X.shape
         total_lens.append(n)
         total_X.append(X)
         total_A.append(A)
         total_D.append(D)
   return {'X':total_X, 'A':total_A, 'L':total_lens, 'F':f, 'D':total_D}


def load_smiles_and_resample(config, smiles_chk):
   raw_data = {'positive':[], 'negative':[]}
   with open(config['positive_data_file'], 'r') as p: 
      raw_data['positive'] = p.readlines()
   
   for neg_name in config['negative_data_files']:
      with open(neg_name, 'r') as n: 
         raw_data['negative'].extend(n.readlines())
   
   positive = np.arange(len(raw_data['positive'])).reshape(-1,1)
   negative = -1 - np.arange(len(raw_data['negative'])).reshape(-1,1)

   x, y, w = __balance_data(positive, negative, config['data_balance'])
   x = x.astype(int)

   smiles_list = []
   smiles_chk = open(smiles_chk, 'w')

   for idx in x[:,0]:
      if idx<0:
         idx = -1 - idx
         flag = 0
         key = 'negative'         
      else:
         flag = 1 
         key = 'positive'
         
      smiles = raw_data[key][idx].strip()
      smiles_list.append(smiles)
      smiles_chk.write('%s %i\n'%(smiles, flag))
      
   smiles_chk.close()

   return smiles_list, y, w


def smiles_to_morgan(smiles, radius, **kwargs):
   mol=Chem.MolFromSmiles(smiles)
   return AllChem.GetMorganFingerprintAsBitVect(mol, radius, **kwargs)


def load_morgan(config_dict, force):
   loader_cfg = config_dict['loader_config']
   data_checkpoint = loader_cfg['data_checkpoint']
   
   if not isfile(data_checkpoint):
      force=True
   if not force:
      return gz_unpickle(data_checkpoint)

   #====== read ======
   smiles_checkpoint = '.'.join(data_checkpoint.split('.')[:-1]+['.smiles'])
   all_smiles, Y, w = load_smiles_and_resample(loader_cfg, smiles_checkpoint)
   output_shape = config_dict['loader_config'].get('output_shape', 1)
   if output_shape==2:
      Y = OneHotEncoder(sparse=False).fit_transform(Y.reshape(-1,1))
   radius = config_dict['vectorizer_config'].get('radius', 2)
   nbits = config_dict['vectorizer_config'].get('nbits', 2048)
   use_features = config_dict['vectorizer_config'].get('use_features', False)
   kwargs = {'nBits':nbits, 'useFeatures':use_features}
   data = np.array([smiles_to_morgan(smiles, radius, **kwargs) for smiles in all_smiles])
   gz_pickle(data_checkpoint, (data, Y, w))

   return data, Y, w


def load_graph_data(config_dict, force):
   loader_cfg = config_dict['loader_config']
   data_checkpoint = loader_cfg['data_checkpoint']
   smiles_checkpoint = data_checkpoint.replace('.pkz','.smiles')
   
   if not isfile(data_checkpoint):
      force=True
   if not force:
      return gz_unpickle(data_checkpoint)
      
   processor = _smiles_data_processor

   #====== read ======
   all_smiles, Y, w = load_smiles_and_resample(loader_cfg, smiles_checkpoint)
   
   #====== processing =======
   tensor_dict = processor(all_smiles, config_dict['vectorization_config'])
   
   #====== joining and padding ============
   max_len_in_data = max(tensor_dict['L'])
   max_len = loader_cfg.get('max_num_atoms', max(tensor_dict['L']))
   assert max_len_in_data<=max_len
   print('Max atom num: ',max_len)
   F = tensor_dict['F']
   lens = tensor_dict['L']

   X = [zero_pad(x, (max_len, F)) for x in tensor_dict['X']]
   A = [zero_pad(x, (max_len, max_len)) for x in tensor_dict['A']]
   D = [zero_pad(x, (max_len, max_len)) for x in tensor_dict['D']]
   
   result = {}
   result['X'] = np.array(X)
   result['A'] = np.array(A)
   result['D'] = np.array(D)
   result['Y'] = np.array(Y)
   result['Y'] = OneHotEncoder(sparse=False).fit_transform(result['Y'].reshape(-1,1))
   result['L'] = np.array(lens)
   raw_chk = data_checkpoint.replace('.pkz','_raw_arrs.pkz')
   gz_pickle(raw_chk, result)
   
   filter_type = config_dict['graph_filters'].get('filter_type', 'first_order')
   adjacency_normalization = config_dict['graph_filters'].get('adjacency_normalization', True)
   filter_generator = FILTERS[filter_type]
   filters = filter_generator(result['A'], result['D'], adjacency_normalization)
   
   #order: X_input, filters_input, nums_input, identity_input, adjacency_input 
   data = result['X'], filters, result['L'], filters[:, :max_len, :], result['A']
   Y = result['Y']
   gz_pickle(data_checkpoint, (data, Y, w))
  
   loader_cfg['max_num_atoms'] = max_len
   return data, Y, w


def _make_indices_from_config(loader_cfg):
   non_zero_idx = loader_cfg.get('non_zero_idx', None)
   data_mu = loader_cfg.get('data_mu', None)
   data_std = loader_cfg.get('data_std', None)
   idx = (data_mu, data_std, non_zero_idx)
   return idx


def load_multitask_clf(config_dict, force):
   loader_cfg = config_dict['loader_config']
   data_checkpoint = loader_cfg['data_checkpoint']
   
   if not isfile(data_checkpoint):
      force=True
   if not force:
      return gz_unpickle(data_checkpoint)
       
   data_file = config_dict['data_file']
   label_file = config_dict['label_file']
   indices = _make_indices_from_config(loader_cfg)
      
   data = gzload(data_file)
   labels = gzload(label_file)
    
   data = _normalize(data, indices, True)
   weights = {}
   Noutputs = labels.shape[1]
   y=[]

   if loader_cfg['kind']=='multitask_clf':
      for i in range(Noutputs):
         name = 'out%i'%i
         w = np.where(labels[:,i]>=0, 1., 0.)
         weights[name]=w
         y.append(np.where(labels[:,i]>0, 1., 0.))
   elif loader_cfg['kind']=='multitask_disengaged':
      output_idx = loader_cfg.get('output_id', 0)
      y = np.where(labels[:,output_idx]>0,1,0)
      mask = np.where(labels[:,i]>=0, 1., 0.)
      data = data[mask]
      y = y[mask]
      weights = compute_sample_weight('balanced', y)
   else:
      raise KeyError('Unknown kind of loader %s'%loader_cfg['kind'])

   gz_pickle(data_checkpoint, (data, y, weights))
    
   return data, y, weights


def load_from_config(config_dict, force, use_idx=True):
   kind = config_dict['loader_config'].get('kind', None)
   if kind is None:
      kind = config_dict['model_params'].get('kind', 'rdkit_ae')
      
   if kind in ['multitask_clf', 'multitask_disengaged']:
      return load_multitask_clf(config_dict, force)
   elif kind in ['ggnn', 'gcnn']:
      return load_graph_data(config_dict, force)
   elif kind=='ggnn_compressed':
      return load_compressed_data(config_dict, force)
   elif kind in ['rbm', 'morgan']:
      return load_morgan(config_dict, force)
   elif not (kind in ['rdkit_ae', 'mold2_ae']):
      raise KeyError('Unknown data preprocessing function')
   
   #PROJECT-SPECIFIC!!
   positive_idx = config_dict.get('drugs_idx', None)
   negative_idx = config_dict.get('non_drugs_idx', None)
   data_list_file = config_dict.get('data_list', None)
   if None in [positive_idx, negative_idx, data_list_file]:
      use_idx = False
   else:
      data_list = load_yaml(data_list_file)
      positive_file = data_list['vectorized_drugs'][positive_idx]
      negative_file = data_list['vectorized_non_drugs'][negative_idx]

   loader_cfg = config_dict['loader_config']
   ##!! mutable !!
   if use_idx:
      loader_cfg['positive_file']=positive_file
      loader_cfg['negative_file']=negative_file
   else:
      positive_file = loader_cfg['positive_file']
      negative_file = loader_cfg['negative_file']
   ##
   data_checkpoint = loader_cfg['data_checkpoint']
   data_balance = loader_cfg['data_balance']
   idx = _make_indices_from_config(loader_cfg)

   return load_data(positive_file, [negative_file], data_checkpoint, data_balance, idx, True, force)


def hu_copy(stratifier, split_indices):
   '''
   Function makes copies of the minority class.
   Done to match results from doi: 10.3389/fgene.2018.00585'''
   
   if type(split_indices).__name__!='list':
      split_indices = list(split_indices)

   selected = stratifier[split_indices]
   Ntotal = len(selected)
   N1 = int(selected.sum())
   N0 = Ntotal-N1

   if N1<N0:
      key_minor, n_major, n_minor = 1, N0, N1
   else:
      key_minor, n_major, n_minor = 0, N1, N0
   
   idx_minor = [x for x in split_indices if stratifier[x]==key_minor]
   
   for x in range(n_major-n_minor):
      x = x%n_minor
      split_indices.append(idx_minor[x])
   
   np.random.shuffle(split_indices)
   
   return list(split_indices)


def split_data(y, weights, cfg_dict, val_mode='val_test', force=False):
   if isfile(cfg_dict['checkpoint']) and not force:
      return gz_unpickle(cfg_dict['checkpoint'])
   
   test_size = cfg_dict.get('test_split', 0.1)
   val_size = cfg_dict.get('validation_split', 0.2)
   random_state = cfg_dict.get('random_state', 123)
   external_test_data = cfg_dict.get('external_test_data', 'none')
   kind = cfg_dict.get('kind', 'binary')
   

   balance_splits = cfg_dict.get('balance_cv_splits', "none") 
   weights_are_None = weights is None
   if weights_are_None or balance_splits!='none':
      weights = np.ones(len(y))

   if kind=='multitask':
      y_to_use=y[0]
      indices = np.arange(len(y[0]))
      kfold_method=KFold
      stratifier = None
   else:
      y_to_use=y
      kfold_method = StratifiedKFold
      indices = np.arange(len(y))
      if len(y.shape)==1:
         stratifier = y
      elif kind in ['binary', 'multiclass']:
         stratifier = y.argmax(axis=1)
      else:
         raise ValueError('Unknown kind of classification. Please select either binary, multiclass or mutlitask')

   split_cfg = {'test_size':test_size, 'random_state':random_state, 'stratify': stratifier}

   if external_test_data=='none':
      train_idx, test_idx = train_test_split(indices, **split_cfg)
   else:
      train_idx, test_idx = indices, []
   
   if balance_splits == 'before':
      train_idx = np.array(hu_copy(stratifier, train_idx)).astype(int)

   if val_mode=='val_test':
      split_cfg['test_size'] = val_size
      split_cfg['stratify'] = stratifier[train_idx] if kind!='multitask' else None
      train_val_splits = [train_test_split(train_idx, **split_cfg)]
   elif val_mode == '5cv_test':
      split_cfg = {'n_splits':5, 'shuffle':True, 'random_state':random_state}
      splitter = kfold_method(**split_cfg)
      train_val_splits = []
      if kind=='multitask':stratifier=np.ones(len(y_to_use))
      for cv_idx in splitter.split(y_to_use[train_idx], stratifier[train_idx]):
         cv_train_idx = train_idx[cv_idx[0]]
         cv_val_idx = train_idx[cv_idx[1]]
         train_val_splits.append((cv_train_idx,cv_val_idx))
   else:
      raise KeyError('Unknown validation mode')
   
   if balance_splits=='after':
      train_val_splits = [tuple(hu_copy(stratifier,x) for x in y) for y in train_val_splits]

   gz_pickle(cfg_dict['checkpoint'], (train_val_splits, train_idx, test_idx, weights))
    
   return train_val_splits, train_idx, test_idx, weights


#TODO refactor; maybe do a recurrent version?
def select_indices(arr_set, indices):
   type_ = type(arr_set).__name__
   if type_ in ['list', 'tuple']:
      sliced = [x[indices] for x in arr_set]
   elif type_=='dict':
      sliced = {}
      for k in arr_set:
         sliced[k] = arr_set[k][indices]
   elif type_=='ndarray':
      sliced = arr_set[indices]
   else:
      raise TypeError('Unrecognized format')
   return sliced


def slice_data(data, indices, num_inputs):
   sliced_data = [select_indices(x, indices) for x in data]
   #if num_inputs==1:
   #   sliced_data = [arr[indices] for arr in data]
   #else:
   #   sliced_data_x = [arr[indices] for arr in data[0]]
   #   sliced_data_rest = [arr[indices] for arr in data[1:]]
   #   sliced_data = [sliced_data_x]+sliced_data_rest
   return sliced_data


def make_test_data(data, indices, num_inputs, config):
   external_test_data = config['cross_validation'].get('external_test_data', 'none')
   if external_test_data == 'none':
      print('no external')
      return slice_data(data, indices, num_inputs)
   config_to_use = deepcopy(config)

   all_prepared = external_test_data.get('all', 'none')
   if all_prepared !='none':
      data = list(gz_unpickle(all_prepared))
      kind = config['model_params'].get('kind','rdkit_ae')
      if kind in ['rdkit_ae','mold2_ae']:
         idx = _make_indices_from_config(config['loader_config'])
         data[0] = _normalize(data[0], idx, True)
      elif kind=='ggnn_compressed':
         data=data[0]
      return data

   
   if config['model_params']['kind'] in ['ggnn', 'gcnn']:
      key_p = '_data_file'
      key_n = key_p+'s'
      neg_val = [external_test_data['negative']]
   elif config['model_params']['kind'] not in ['multitask_clf', 'multitask_disengaged', 'multitask']:
      key_p = '_file'
      key_n = key_p
      neg_val = external_test_data['negative']

   if config['model_params']['kind'] in ['multitask_clf', 'multitask_disengaged', 'multitask']:
      config_to_use['loader_config']['data_file'] = external_test_data['data_file']
      config_to_use['loader_config']['label_file'] = external_test_data['label_file']
   else:
      config_to_use['loader_config']['positive'+key_p] = external_test_data['positive']
      config_to_use['loader_config']['negative'+key_n] = neg_val
   
   config_to_use['loader_config']['data_checkpoint'] = \
      config['loader_config']['data_checkpoint'].replace('chk','test_chk')

   return load_from_config(config_to_use, external_test_data['force'], use_idx=False)


def to_binary(labels):
   shape = np.shape(labels)
   Ndim = len(shape)
   assert Ndim<=2, 'Up to 2D arrays supported, got %i'%Ndim
   
   if Ndim==1: #binary
      to_use = labels
   elif shape[1]==1:
      to_use = labels[:,1]
   else:
      to_use = labels.argmax(axis=1)
      
   return to_use


def scale_weights(labels, weights, scaling_factor=0.5):
   '''positive class assumed to be 1'''
   to_use = to_binary(labels)
   assert scaling_factor<1.0 and scaling_factor>0.0
   
   Nsample = np.shape(labels)[0]
   result = np.ones(Nsample)
   if not(weights is None) and scaling_factor!=0.5:
      positive_class = to_use==1
      other = to_use!=1
      result *= weights
      result[positive_class] = result[positive_class]*scaling_factor
      result[other] = result[other]*(1-scaling_factor)
   elif not(weights is None):
      result = weights

   return result


def tpr_tnr(y_pred, y_true):
   pred = to_binary(y_pred).round(0)
   true = to_binary(y_true)
   Npositive = sum(true)
   Nnegative = sum(1-true)
   true_positive = sum(true*pred)
   true_negative = sum((1-true)*(1-pred))
      
   return true_positive/Npositive, true_negative/Nnegative

