import argparse
from myutils.formatter import PrettyFormatter

parser=argparse.ArgumentParser(formatter_class=PrettyFormatter)
#
parser.add_argument('--output_core', type=str, default='experiments/def_result',
   help='Core part of the output files (in the form: output_core_appendix)')
#
parser.add_argument('--splits_checkpoint', type=str, default=None,
   help=('Checkpoint containing indeces of train/test splits. '
         'If present, will overriden splits generation and try to read from this file.'))
#
parser.add_argument('--data_checkpoint', type=str, default=None, 
   help=('Checkpoint containing preprocessed dataset. '
         'If present, will override preprocessing step and try to read from this file.'))
#
parser.add_argument('--scale_positive', type=float, default=0.5, 
   help=('Scale assigned to positive class. Value of 0.5 means that both negative and'
        ' positive classes are treated equally.'))
#
parser.add_argument('--config', type=str, default='config_files/def_rdkit_config.yaml', 
   help='YAML file defineing exeriment documentation (see README)')
#
parser.add_argument('--force', type=str, choices=['all','splits'], default=None, 
   help=('Forces data preprocess/splits calculation despite checkpoint being present.\n'
         '* all - force both data prerocess and splits generation\n'
         '* splits - force only splits generation\n'))
#
parser.add_argument('--epochs', type=int, default=100,
   help='Number of training epochs') 
#
parser.add_argument('--validation_mode', type=str, choices=['5cv_test','val_test'], 
   default='val_test', help=('Type of validation procedure. Validation score is '
   'used for model selection, which is than tested on the held-out test set.\n'
   '* val_test - train on 72%% of data, validate on 18%% of data and test on 10%% of data\n'
   '* 5cv_test - do 5-fold cross-validation on 90%% of data, test on 10%%%% of data\n'))
#
args=parser.parse_args()

#name checkoints of this run (if checkpoint file is not provided
if args.data_checkpoint is None:
   args.data_checkpoint = '%s_data_chk.pkz'%args.output_core

if args.splits_checkpoint is None:
   args.splits_checkpoint = '%s_splits_chk.pkz'%args.output_core

#set override flags; preprocessing and splits will be done,
#results will be written in checkpoints
args.force_resample=False
args.force_resplit=False
if args.force=='all':
   args.force_resample=True
   args.force_resplit=True
elif args.force=='splits':
   args.force_resplit=True

#===== These imports take time, so are placed here. Thanks to that printing help is quick
from data_preprocess import load_yaml, load_from_config, yaml, gz_pickle, gz_unpickle, tpr_tnr
from data_preprocess import balance_masked_weights, split_data, isfile, np, slice_data, make_test_data, scale_weights
from models import make_model, make_av_std, make_bayesian_prediction, SaveSelected, tasks_balanced_scores
import logging

#========= logging stuff ===========
logger=logging.getLogger()
level=logging.INFO
formatter=logging.Formatter('%(asctime)s: %(levelname)s): %(message)s')
log_name = '%s_%s.log'%(args.output_core, args.validation_mode)
handlers= logging.FileHandler(log_name), logging.StreamHandler()
for handler in handlers:
   handler.setFormatter(formatter)
   handler.setLevel(level)
   logger.addHandler(handler)
logger.setLevel(level)

#======= loading stuff ==========
logger.info('Start')
config = load_yaml(args.config)
config['loader_config']['data_checkpoint'] = args.data_checkpoint
x, y, weights = load_from_config(config, args.force_resample)

#======= split data =======
checkpoint_name = args.splits_checkpoint
cv_cfg = config['cross_validation']
cv_cfg['checkpoint'] = checkpoint_name
cv_splits, train_idx, test_idx, weights = split_data(y, weights, cv_cfg, 
                                   args.validation_mode, args.force_resplit)
data_stuff = [x, y, weights]

kind = config['model_params'].get('kind', 'any')
is_multitask = 'multitask' in kind
if is_multitask:
   axis = 1
   N_outputs = len(y)
else:
   axis=0
   N_outputs = -1

to_relax = config['model_params'].get('relax', False)

#======= print info ======= 
config_str = yaml.dump(config, default_flow_style=False)
logger.info('Config file: %s'%args.config)
logger.info('Force resample/reload: %s   Force resplit: %s'%(args.force_resample, args.force_resplit))
logger.info('weights_are_none %s'%(weights is None))
logger.info('Validation mode: %s'%args.validation_mode)
logger.info('output_core: %s'%args.output_core)
logger.info('Epochs: %i'%args.epochs)
logger.info('Positive scaling: %8.3f'%args.scale_positive)
logger.info('Data loaded.\n== Config ==\n%s\n============'%config_str)
data_summary = [str(obj) for obj in [np.shape(y), np.sum(y, axis=axis)]]
logger.info('Y shape:%s  Y sum:%s'%tuple(data_summary))

#write all configuration to the YAML file
with open('%s_post.yaml'%args.output_core, 'w') as f:
   f.write(config_str)

#====== train =======

batch_size=100 ## add config?? later??
epochs=args.epochs
history=[]

N_inputs = config['model_params'].get('num_inputs', 1)
if N_inputs==1:
   input_shape = x.shape[1]
else:
   input_shape = [arr.shape for arr in x]

data_shapes = config.get('data_shapes', 'none')
if data_shapes!='none':
   input_shape = data_shapes

for cv_train_idx, cv_val_idx in cv_splits:
   train = slice_data(data_stuff, cv_train_idx, N_inputs)
   val = slice_data(data_stuff, cv_val_idx, N_inputs)
   if is_multitask:
      logger.info('Train Y: %s'%str(np.sum(train[1],axis=1)))
      for i,vy in enumerate(train[1]):
         key='out%i'%i
         train[2][key] = balance_masked_weights(vy,train[2][key])
         val[2][key] = balance_masked_weights(val[1][i],val[2][key])
   else:
      train[2] = scale_weights(train[1], train[2], args.scale_positive)
      val[2] = scale_weights(val[1], val[2], args.scale_positive)
      logger.info('Train Y: %s'%str(np.sum(train[1],axis=0)))
   model, metric = make_model(input_shape, np.shape(y), config['model_params']) 
   logger.info('Model build')
   result = model.fit(train[0], train[1], sample_weight = train[2],
                        validation_data = val,
                        batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)
   if to_relax:
      logger.info('Relaxing')
      tr_score, _ = make_av_std([result.history], metric, is_multitask, N_outputs)
      tst_score, _ =  make_av_std([result.history], 'val_%s'%metric, is_multitask, N_outputs)
      logger.info('Last stats: Train:  %8.3f   Val: %8.3f'%(tr_score.max(), tst_score.max()))
      w= model.get_weights()
      model, _ = make_model(input_shape, np.shape(y), config['model_params'])
      model.set_weights(w)
      for layer in model.layers: layer.trainable=True
      result = model.fit(train[0], train[1], sample_weight = train[2],
                         validation_data = val[:2],
                         batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)
      tr_score, _ = make_av_std([result.history], metric, is_multitask, N_outputs)
      tst_score, _ =  make_av_std([result.history], 'val_%s'%metric, is_multitask, N_outputs)
      logger.info('After relaxation: Train:  %8.3f   Val: %8.3f'%(tr_score.max(), tst_score.max()))
   history.append(result.history)

#====== Average ===============
if is_multitask:
   metric = 'out%i_'+metric
cv_av, cv_std = make_av_std(history, metric, is_multitask, N_outputs)
cv_val_av, cv_val_std = make_av_std(history, 'val_%s'%metric, is_multitask, N_outputs)
val_loss = np.array([x['val_loss'] for x in history]).mean(axis=0)
if is_multitask:
   best_cv_idx = np.argmax(val_loss)#cv_val_av.argmax()
else:
   best_cv_idx = cv_val_av.argmax()

logging.info('Val loss: %i-%8.6f'%(best_cv_idx,val_loss[best_cv_idx]))
test_epochs = args.epochs #best_cv_idx+1

#====== test ===========
logger.info('Testing')
test_stuff = make_test_data(data_stuff, test_idx, N_inputs, config)#slice_data(data_stuff, test_idx, N_inputs)
train_stuff = slice_data(data_stuff, train_idx, N_inputs)

test_stuff = list(test_stuff)
train_stuff = list(train_stuff)

#TODO: repetition-refactor
if is_multitask:
   for i,vy in enumerate(train_stuff[1]):
      key='out%i'%i
      train_stuff[2][key] = balance_masked_weights(vy, train_stuff[2][key])
      test_stuff[2][key] = balance_masked_weights(test_stuff[1][i], test_stuff[2][key])
else:
      train_stuff[2] = scale_weights(train_stuff[1], train_stuff[2], args.scale_positive)
      test_stuff[2] = scale_weights(test_stuff[1], test_stuff[2], args.scale_positive)

model, _ = make_model(input_shape, np.shape(y), config['model_params']) 
logger.info('Model build')
saver = SaveSelected(best_cv_idx)
try:
   result = model.fit(train_stuff[0], train_stuff[1], sample_weight=train_stuff[2],
                   validation_data = test_stuff,
                   batch_size=batch_size, epochs=test_epochs, shuffle=True, verbose=1, callbacks=[saver])
   if to_relax:
      logger.info('Relaxing')
      tr_score, _ = make_av_std([result.history], metric, is_multitask, N_outputs)
      tst_score, _ =  make_av_std([result.history], 'val_%s'%metric, is_multitask, N_outputs)
      logger.info('Last stats: Train:  %8.3f   Val: %8.3f'%(tr_score.max(), tst_score.max()))
      for layer in model.layers: layer.trainable=True
      result = model.fit(train_stuff[0], train_stuff[1], sample_weight=train_stuff[2],
                   validation_data = test_stuff[:2],
                   batch_size=batch_size, epochs=test_epochs, shuffle=True, verbose=1, callbacks=[saver])

      tr_score, _ = make_av_std([result.history], metric, is_multitask, N_outputs)
      tst_score, _ =  make_av_std([result.history], 'val_%s'%metric, is_multitask, N_outputs)
      logger.info('After relaxation: Train:  %8.3f   Val: %8.3f'%(tr_score.max(), tst_score.max()))
except:
   xx = list(test_stuff[0])
   yy = test_stuff[1]
   result = model.fit(train_stuff[0], train_stuff[1], sample_weight=train_stuff[2],
                       validation_data = (xx, yy, test_stuff[2]),
                       batch_size=batch_size, epochs=test_epochs, shuffle=True, verbose=1, callbacks=[saver])
   if to_relax:
      logger.info('Relaxing')
      tr_score, _ = make_av_std([result.history], metric, is_multitask, N_outputs)
      tst_score, _ =  make_av_std([result.history], 'val_%s'%metric, is_multitask, N_outputs)
      logger.info('Last stats: Train:  %8.3f   Val: %8.3f'%(tr_score.max(), tst_score.max()))
      for layer in model.layers: layer.trainable=True
      result = model.fit(train_stuff[0], train_stuff[1], sample_weight=train_stuff[2],
                         validation_data = (xx, yy),
                         batch_size=batch_size, epochs=test_epochs, shuffle=True, verbose=1, callbacks=[saver])

#====== save history =====
logger.info('Saving history...')
test_av, test_std = make_av_std([result.history], metric, is_multitask, N_outputs)
test_val_av, test_val_std = make_av_std([result.history], 'val_%s'%metric, is_multitask, N_outputs)

logger.info('Best CV:')
logger.info('epoch: %i'%(best_cv_idx+1))
cv_max_data = tuple(arr[best_cv_idx] for arr in [cv_av, cv_std, cv_val_av, cv_val_std])
logger.info('train: %8.3f (%4.3f)   val:%8.3f (%4.3f)'%cv_max_data)
logger.info('Test:')
test_max_data = tuple(arr[best_cv_idx] for arr in [test_av, test_val_av])
logger.info('train: %8.3f   val:%8.3f'%test_max_data)

logger.info('Resetting model weights from epoch with best cross-val score')
saver.reset()

if is_multitask:
   logger.info('Details at best epoch:')
   for i in range(N_outputs):
      out_trn = result.history[metric%i][best_cv_idx]
      out_tst = result.history['val_'+metric%i][best_cv_idx]
      logger.info('   out%-2i:  train: %8.3f  test %8.3f'%(i, out_trn, out_tst))
else:
   if type(test_stuff[0]).__name__=='tuple': 
      x_to_use = list(test_stuff[0])
   else:
      x_to_use = test_stuff[0]
   pred = model.predict(x_to_use)
   tpr, tnr = tpr_tnr(pred, test_stuff[1])
   logger.info('TPR: %8.3f  TNR: %8.3f'%(tpr, tnr))

with open('%s_%s_history.txt'%(args.output_core, args.validation_mode), 'w') as f:
   data_order = [cv_av, cv_std, cv_val_av, cv_val_std, test_av, test_val_av]
   titles = ('EPOCH', 'train_cv', 'train_cv_err', 'val_cv', 'val_cv_err', \
                      'test', 'test_val')

   line_format = '%5i %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f\n'

   f.write('%5s %12s %12s %12s %12s %12s %12s\n'%titles)
   for i in range(args.epochs):
      to_write = [i+1] + [arr[i] for arr in data_order]
      f.write(line_format%tuple(to_write))


if is_multitask:
   prediction_train = model.predict(train_stuff[0])
   prediction_test = model.predict(test_stuff[0])
   score_train = tasks_balanced_scores(prediction_train, train_stuff[1], train_stuff[2])
   score_test = tasks_balanced_scores(prediction_test, test_stuff[1], test_stuff[2])
   auc_test = tasks_balanced_scores(prediction_test, test_stuff[1], test_stuff[2], auc=True)
   logger.info('Correcting for masking information')
   logger.info('Train mean: %8.3f  Test mean: %8.3f  Test mean auc %8.3f'%(score_train.mean(), score_test.mean(), np.mean(auc_test)))
   for i in range(N_outputs):
      logger.info('   out%-2i:  train: %8.3f  test %8.3f auc: %8.3f'%(i, score_train[i], score_test[i], auc_test[i]))


logger.info('Saving model weights')
gz_pickle(args.output_core+'_train_weights.pkz', model.get_weights())

#======= bayesian prediction =======

do_bayes = config['model_params'].get('dropout_flag',False)

if do_bayes:
   logger.info('Making bayesian prediction on %i samples'%test_stuff[1].shape[0])
   bayesian_result = make_bayesian_prediction(model, test_stuff[0])
   
   logger.info('Done. Saving...')
   with open('%s_%s_bayesian_result.txt'%(args.output_core, args.validation_mode), 'w') as f:
      order = ('true_flag','mean_prob', 'aleatoric', 'epistemic')
      f.write('%10s %10s %10s %10s \n'%order)
   
      for i in range(test_stuff[1].shape[0]):
         f.write('%10i '%test_stuff[1][i].argmax())
   
         for key in order[1:]:
            f.write('%10.4f '%bayesian_result[key][i])
         
         f.write('\n')
         
logger.info('EOT, NCR')

