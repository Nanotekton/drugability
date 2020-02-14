import argparse
from myutils.formatter import PrettyFormatter

parser=argparse.ArgumentParser(formatter_class=PrettyFormatter)
#
parser.add_argument('--output_core', type=str, default='experiments/pu_fuselier_result',
   help='Core part of the output files (in the form: output_core_appendix)')
#
parser.add_argument('--data_checkpoint', type=str, default=None, 
   help=('Checkpoint containing preprocessed dataset. '
         'If present, will override preprocessing step and try to read from this file.'))
#
parser.add_argument('--config', type=str, default='config_files/def_rdkit_config.yaml', 
   help='YAML file defineing exeriment documentation (see README)')
#
parser.add_argument('--reload', action='store_true',  
   help='Forces data preprocess despite checkpoint being present.')
#
parser.add_argument('--epochs', type=int, default=100,
   help='Number of training epochs') 
#
parser.add_argument('--use_subset_first', action='store_true',
   help=('During first iteration, take only subset of the negative set. '
         "Since it's bigger, it will save some comutational time"))
#
parser.add_argument('--th', type=float, default=0.5,
   help='Maximum drug-likeliness of compound to be reasonable "non-drug-like"')
#
parser.add_argument('--nproc', type=int, default=1,
   help='Number of threads for cross-validation')
#
args=parser.parse_args()


#====================
from data_preprocess import load_yaml, load_from_config, yaml, gz_pickle, \
                            split_data, gz_unpickle, isfile, np, slice_data, make_test_data, StratifiedKFold
from models import make_model, make_av_std, make_bayesian_prediction, SaveSelected
import logging
from multiprocessing import Pool

#========= logging stuff ===========
logger=logging.getLogger()
level=logging.INFO
formatter=logging.Formatter('%(asctime)s: %(levelname)s): %(message)s')
log_name = '%s.log'%(args.output_core)
handlers= logging.FileHandler(log_name), logging.StreamHandler()
for handler in handlers:
   handler.setFormatter(formatter)
   handler.setLevel(level)
   logger.addHandler(handler)
logger.setLevel(level)

#======= loading stuff ==========
logger.info('Start')
config = load_yaml(args.config)
if args.data_checkpoint is None:
   args.data_checkpoint = args.output_core + '_data_chk.npz'
config['loader_config']['data_checkpoint'] = args.data_checkpoint
config['training_cfg'] = dict(batch_size=100, epochs=args.epochs, shuffle=True, verbose=0)
x, y, weights = load_from_config(config, args.reload)

#!!!! QUICK FIX!!!!
x=np.nan_to_num(x)
x = np.where(abs(x)<10.0, x, 0)

data_stuff = [x, y, weights]

if len(y.shape)==2:
   assert y.shape[1]==2, "Y should be binary, y.shape=%s"%str(y.shape)
   positive_idx = np.where(y.argmax(axis=1)==1)[0]
   unlabelled_idx = np.where(y.argmax(axis=1)==0)[0]
   negative_idx = np.where(y.argmax(axis=1)==0)[0]
elif len(y.shape)==1:
   positive_idx = np.where(y==1)[0]
   unlabelled_idx = np.where(y==0)[0]
   negative_idx = np.where(y==0)[0]
else:
   raise ValueError('Incorrect shape, y.shape='%str(y.shape))

if args.use_subset_first:
   indices_to_use = np.append(positive_idx, negative_idx[:len(positive_idx)])
   weights = np.ones(y.shape[0])
else:
   indices_to_use = np.append(positive_idx, negative_idx)
#======= print info ======= 
config_str = yaml.dump(config, default_flow_style=False)
logger.info('Config file: %s'%args.config)
logger.info('Force resample/reload: %s   '%(args.reload))
logger.info('Restrict first iteration negatives: %s   '%(args.use_subset_first))
logger.info('weights_are_none %s'%(weights is None))
logger.info('output_core: %s'%args.output_core)
logger.info('Epochs: %i'%args.epochs)
logger.info('RN th: %f'%args.th)
logger.info('Data loaded.\n== Config ==\n%s\n============'%config_str)
data_summary = [str(obj) for obj in [y.shape, y.sum(axis=0)]]
logger.info('Y shape:%s  Y sum:%s'%tuple(data_summary))

with open('%s_post.yaml'%args.output_core, 'w') as f:
   f.write(config_str)

#====== train =======

#5cv split with seed=123; fixed for now
def pu_cv_splitter(indices_to_use, y):
   y_to_use = y[indices_to_use]
   if len(y_to_use.shape)==2:
      y_to_use=y_to_use.argmax(axis=1)
   splitter = StratifiedKFold(n_splits=5, random_state=123)
   for train_idx_internal, test_idx_internal in splitter.split(y_to_use, y_to_use):
      train_idx = indices_to_use[train_idx_internal]
      val_idx = indices_to_use[test_idx_internal]
      yield train_idx, val_idx


def get_input_shapes(data_stuff, config):
   N_inputs = config['model_params'].get('num_inputs', 1)
   if N_inputs==1:
      input_shape = data_stuff[0].shape[1]
   else:
      input_shape = [arr.shape for arr in data_stuff[0]]
   
   data_shapes = config.get('data_shapes', 'none')
   if data_shapes!='none':
      input_shape = data_shapes
   return N_inputs, input_shape  


def make_cv_on_current_set(data_stuff, indices_to_use, config):
   '''On current set, perform cv and extract model, score and best iteration '''
   history=[]
   N_inputs, input_shape = get_input_shapes(data_stuff, config)
   
   #for cv_train_idx, cv_val_idx in pu_cv_splitter(indices_to_use, data_stuff[1]):
   def compute_fold(ARG): 
      cv_train_idx, cv_val_idx  = ARG 
      train = slice_data(data_stuff, cv_train_idx, N_inputs)
      val = slice_data(data_stuff, cv_val_idx, N_inputs)
      model, metric = make_model(input_shape, data_stuff[1].shape, config['model_params']) 
      logger.info('Model build')
      result = model.fit(train[0], train[1], sample_weight = train[2],
                        validation_data = val[:2], **config['training_cfg'])
      return result.history, metric 
   
   #   history.append(result.history)
    
   splits = pu_cv_splitter(indices_to_use, data_stuff[1]) 
   if args.nproc <=1:
      history = [compute_fold(ARG) for ARG in splits]
   else:
      p=Pool(args.nproc)
      history = p.map(compute_fold, splits) 
      del p

   metric = history[0][1]
   history = [ARG[0] for ARG in history]

   #====== Average ===============
   cv_av, cv_std = make_av_std(history, metric)
   cv_val_av, cv_val_std = make_av_std(history, 'val_%s'%metric)
   best_cv_idx = cv_val_av.argmax()
    
   #====== test ===========
   logger.info('Making Final model')
   train_stuff = slice_data(data_stuff, indices_to_use, N_inputs)
   model, metric = make_model(input_shape, data_stuff[1].shape, config['model_params']) 
   logger.info('Model build')
   saver = SaveSelected(best_cv_idx)
   result = model.fit(train_stuff[0], train_stuff[1], sample_weight=train_stuff[2],
                      callbacks=[saver], **config['training_cfg'])
   saver.reset()
   
   return {'train': (cv_av, cv_std), 'val':(cv_val_av, cv_val_std),
           'it':best_cv_idx, 'model':model}


def make_stats_from_vector(vector):
   assert len(vector.shape)==1, 'should be 1D'
   min_, max_ = vector.min(), vector.max()
   N = len(vector)
   if N<=12:
      av = vector.mean()
      std = vector.std()
      p1, p2 = av-std, av+std
   else:
      mid = int(N/2)
      q1, q2 = int(0.25*N), int(0.75*N)
      v = np.sort(vector)
      av = v[mid]
      p1, p2 = v[q1], v[q2]
   return min_, p1, av, p2, max_


def extract_reliable_negatives_fuselier(model, data_stuff, negative_idx, th=0.5, yshape=2, N_inputs=1):
   negative_stuff = slice_data(data_stuff, negative_idx, N_inputs)
   drugability = model.predict(negative_stuff[0])
   if yshape==2:
      drugability=drugability[:,1]
   new_reliable_negatives_internal = np.where(drugability<th)[0]
   new_reliable_negatives = negative_idx[new_reliable_negatives_internal]

   return new_reliable_negatives, make_stats_from_vector(drugability), np.where(drugability>0.5,1,0).sum()


def get_new_weights(y_all, reliable_negatives):
   binary_elements_condition = set(y_all.reshape(-1)) in [{0,1}, {0.0, 1.0}]
   two_class_condition = len(y_all.shape)<=2
   message = 'Y should be binary, y.shape=%s'%str(y_all.shape)
   assert binary_elements_condition and two_class_condition, message

   if len(y_all.shape)==2:
      y_to_use = y.argmax(axis=1)
   else:
      y_to_use=y_all
   
   N_positive = y_to_use.sum()
   N_negative = len(reliable_negatives)
   N_tot = float(N_positive+N_negative)
   w_pos = N_negative/N_tot
   w_neg = N_positive/N_tot
   
   weights = np.where(y_to_use==1, w_pos, w_neg)
   
   return weights


#==== main loop ========
Nu, Npos = data_stuff[1].sum(axis=0)
new_negatives = Nu
prev_negatives= Nu+1
iteration = 1
maxit=100
while new_negatives<prev_negatives and iteration<maxit and new_negatives>Npos:
   logging.info('PU iter: %i'%iteration)
   cv_results = make_cv_on_current_set(data_stuff , indices_to_use, config)
   best_it = cv_results['it']
   score_at_best = '%5i '%best_it
   for k in ['train', 'val']:
      v = cv_results[k]
      score_at_best+= '%s %8.3f (%.3f) '%(k, v[0][best_it], v[1][best_it])
   logging.info("Best CV iter: {:s}".format(score_at_best))
   
   #update reliable
   negative_idx, stats, maybe_drug = extract_reliable_negatives_fuselier(cv_results['model'], data_stuff, negative_idx, th=args.th)
   prev_negatives = new_negatives
   new_negatives = len(negative_idx)

   #update lists
   indices_to_use = np.append(positive_idx, negative_idx)
   #update weights for next iteration
   data_stuff[2] = get_new_weights(data_stuff[1], negative_idx)

   N_neg, N_unlabelled = new_negatives, Nu-new_negatives
   logging.info('Reliable negatives: %i, rejected data: %i'%(N_neg, N_unlabelled))
   logging.info('Unlabelled drugability stats: '+ ' '.join(['%5.3f '%dg for dg in stats]))
   logging.info('May-be drugs: %i'%maybe_drug)
   iteration+=1
   if iteration==maxit:
      logging.info('Maximum number of iteration achieved')

#save model and labels
logging.info('Saving last indices and weigths')
gz_pickle(args.output_core+'_indices_and_weights.pkz',\
          {'model_weights': cv_results['model'].get_weights(),
           'negative': negative_idx,
           'unlabelled': unlabelled_idx})

logger.info('EOT, NCR')
