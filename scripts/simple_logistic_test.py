import argparse
from myutils.formatter import PrettyFormatter

parser=argparse.ArgumentParser(formatter_class=PrettyFormatter,\
   desc='Copy of "make_experiment.py" adjusted for sklearn model API')
#
parser.add_argument('--output_core', type=str, default='experiments/def_result',
   help='Core part of the output files (in the form: output_core_appendix)')
#
parser.add_argument('--config', type=str, default='config_files/def_rdkit_config.yaml', 
   help='YAML file defineing exeriment documentation (see README)')
#
parser.add_argument('--force', type=str, choices=['all','splits'], default=None, 
   help=('Forces data preprocess/splits calculation despite checkpoint being present.\n'
         '* all - force both data prerocess and splits generation\n'
         '* splits - force only splits generation\n'))
#
parser.add_argument('--validation_mode', type=str, choices=['5cv_test','val_test'], 
   default='val_test', help=('Type of validation procedure. Validation score is '
   'used for model selection, which is than tested on the held-out test set.\n'
   '* val_test - train on 72%% of data, validate on 18%% of data and test on 10%% of data\n'
   '* 5cv_test - do 5-fold cross-validation on 90%% of data, test on 10%%%% of data\n'))
#
args=parser.parse_args()

args.force_resample=False
args.force_resplit=False
if args.force=='all':
   args.force_resample=True
   args.force_resplit=True
elif args.force=='splits':
   args.force_resplit=True

#====================
from data_preprocess import load_yaml, load_from_config, yaml, gz_pickle, \
                            split_data, gz_unpickle, isfile, np, slice_data, make_test_data

from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import logging

#========= logging stuff ===========
logger=logging.getLogger()
level=logging.INFO
formatter=logging.Formatter('%(asctime)s: %(levelname)s): %(message)s')
log_name = '%s_%s.log'%(args.output_core, args.validation_mode)
handlers = logging.FileHandler(log_name), logging.StreamHandler()
for handler in handlers:
   handler.setFormatter(formatter)
   handler.setLevel(level)
   logger.addHandler(handler)
logger.setLevel(level)

#======= loading stuff ==========
logger.info('Start')
config = load_yaml(args.config)
config['loader_config']['data_checkpoint'] = 'logistic_data_chk.pkz'
x, y, weights = load_from_config(config, args.force_resample)
logging.info('X shape: %s'%str(x.shape))
x = x[:,-1].reshape(-1,1)
#======= split data =======
checkpoint_name = 'logistic_splits_chk.pkz'
cv_cfg = config['cross_validation']
cv_cfg['checkpoint'] = checkpoint_name
cv_splits, train_idx, test_idx, weights = split_data(y, weights, cv_cfg, 
                                   args.validation_mode, args.force_resplit)
data_stuff = [x, y, weights]

#======= print info ======= 
config_str = yaml.dump(config, default_flow_style=False)
logger.info('Config file: %s'%args.config)
logger.info('Force resample/reload: %s   Force resplit: %s'%(args.force_resample, args.force_resplit))
logger.info('weights_are_none %s'%(weights is None))
logger.info('Validation mode: %s'%args.validation_mode)
logger.info('output_core: %s'%args.output_core)
logger.info('Data loaded.\n== Config ==\n%s\n============'%config_str)
data_summary = [str(obj) for obj in [y.shape, y.sum(axis=0)]]
logger.info('Y shape:%s  Y sum:%s'%tuple(data_summary))

with open('%s_post.yaml'%args.output_core, 'w') as f:
   f.write(config_str)

#====== train =======


def create_model(input_shape, output_shape, model_config):
   hidden_units = model_config.get('hidden_units', 128)
   C = model_config.get('C',100)
   logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
   logistic.C = C
   return logistic

N_inputs=1
history=[]
for cv_train_idx, cv_val_idx in cv_splits:
   train = slice_data(data_stuff, cv_train_idx, N_inputs)
   val = slice_data(data_stuff, cv_val_idx, N_inputs)
   model = create_model(None, None, config['model_params']) 
   logger.info('Model build')
   y = train[1]
   if len(y.shape)==2:
      y=y.argmax(axis=1)
   yval = val[1]
   if len(yval.shape)==2:
      yval=yval.argmax(axis=1)
   model.fit(train[0], y)
   pred = model.predict(val[0])
   precision = metrics.precision_score(yval, pred)
   recall = metrics.recall_score(yval, pred)
   bacc = metrics.balanced_accuracy_score(yval, pred)
   logger.info('B_accuracy: %8.3f  precision:%8.3f  recall: %8.3f'%(bacc, precision, recall))
   history.append([bacc, precision, recall])

mean=np.mean(history, axis=0)
std = np.std(history, axis=0)/np.sqrt(max(1,len(cv_splits)-1))
logger.info('Mean B_accuracy: %8.3f  precision:%8.3f  recall: %8.3f'%tuple(mean))
logger.info(' STD B_accuracy: %8.3f  precision:%8.3f  recall: %8.3f'%tuple(std))

#====== test ===========
logger.info('Testing')
test_stuff = make_test_data(data_stuff, test_idx, N_inputs, config)#slice_data(data_stuff, test_idx, N_inputs)
train_stuff = slice_data(data_stuff, train_idx, N_inputs)
model = create_model(None, y.shape, config['model_params']) 
logger.info('Model built')
ytrain, ytest = train_stuff[1], test_stuff[1]
if len(ytrain.shape)==2:
   ytrain = ytrain.argmax(axis=1)
   ytest = ytest.argmax(axis=1)

model.fit(train_stuff[0], ytrain)
pred = model.predict(test_stuff[0])
precision = metrics.precision_score(ytest, pred)
recall = metrics.recall_score(ytest, pred)
bacc = metrics.balanced_accuracy_score(ytest, pred)
logger.info('Test:')
logger.info('B_accuracy: %8.3f  precision:%8.3f  recall: %8.3f'%(bacc, precision, recall))

logger.info('EOT, NCR')
