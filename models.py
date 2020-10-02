from keras.models import Model
from keras.layers import Input, Dense, Dropout, RepeatVector, Lambda, Activation, BatchNormalization, Reshape, Flatten
from keras.engine.topology import Layer
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import Callback
from keras_dgl.layers import MultiGraphCNN, MultiGraphAttentionCNN, GraphConvLSTM
from data_preprocess import isfile, gz_unpickle, np
from balance import balanced_accuracy, balanced_categorical_accuracy
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
#from myutils.graph_compression import tf_bnd_to_adj, tf
import tensorflow as tf
balanced_metrics = {'balanced_accuracy':balanced_accuracy, 'balanced_categorical_accuracy':balanced_categorical_accuracy}
balanced_metrics['balanced_acc'] = balanced_metrics['balanced_accuracy']

def masked_balanced_score(y_pred, y_true, weights, balance=True, auc=False):
   w = np.where(weights>0)[0]
   if balance:
      function = balanced_accuracy_score
   else:
      function = accuracy_score
   if auc:
      function = roc_auc_score
   to_use_pred = y_pred[w]
   to_use_true = y_true[w]
   return function(to_use_true, to_use_pred)


def tasks_balanced_scores(y_pred, y_true, weights, balance=True, auc=False):
   result = []
   for i, y_p in enumerate(y_pred):
      y_t = y_true[i]
      w = weights['out%i'%i]
      assert len(w)==len(y_t)
      assert len(w)==len(y_p)
      if not auc:
         to_use = y_p.round(0)
      else:
         to_use = y_p
      result.append(masked_balanced_score(to_use, y_t, w, balance=balance, auc=auc))
   return np.array(result)


class SaveSelected(Callback):
    def __init__(self, target=-1):
        super().__init__()
        self.target=target

    def on_epoch_end(self, epoch, logs=None):
        if epoch==self.target:
             self.weights = self.model.get_weights()
     
    def reset(self):
        weights = getattr(self, 'weights', None)
        if not (weights is None):
           self.model.set_weights(weights)


def make_ae_model(input_shape, output_shape, model_config):
   input_layer = Input(shape=(input_shape,))
   if len(output_shape)==1:
      output_activation = 'sigmoid' 
      loss='binary_crossentropy'
      metric='acc'
      output_shape=1
   else:
      output_activation = 'softmax'
      loss = 'categorical_crossentropy'
      metric = 'categorical_accuracy'
      output_shape=2
    
   #get params
   balance_metric = model_config.get('metric_balance', False)
   if balance_metric:
      metric = 'balanced_'+metric
      if metric=='balanced_acc': metric = 'balanced_accuracy'
      metric_f = balanced_metrics[metric]
   else:
      metric_f = metric

   drp = model_config.get('model_dropout', 0.2)
   drp_flag = model_config.get('dropout_flag', False)
   l2_val = model_config.get('model_l2', 1e-3)
   activation = model_config.get('encoder_hidden_activation', 'relu')
   init_weights_file = model_config.get('encoder_init_weights', None)
   lr = model_config.get('model_lr', 1e-3)
   freeze_encoder = model_config.get('freeze_encoder', False)

   n_hidden = model_config.get('encoder_hidden_units', 90)
   n_layers = model_config.get('encoder_num_layers', 1)
   if type(n_hidden).__name__ == 'int':
      n_hidden = [n_hidden for _ in range(n_layers)]
   #make model
   hidden = Dropout(drp)(input_layer, training=drp_flag)
   for N in n_hidden:
      hidden = Dense(N, activation=activation, 
                     kernel_regularizer=l2(l2_val))(hidden)
      hidden = Dropout(drp)(hidden, training=drp_flag)
   
   encoder = Model(inputs=input_layer, outputs=hidden)

   output = Dense(output_shape, activation=output_activation, 
                  kernel_regularizer=l2(l2_val))(hidden)
    
   model = Model(inputs=input_layer, outputs=output)

   #assign weights
   init_available = not( init_weights_file is None)
   if init_available:
      init_available = isfile(init_weights_file)
    
   if init_available:
      weights = gz_unpickle(init_weights_file)
      encoder.set_weights(weights)
   if freeze_encoder:
      for layer in encoder.layers:
         layer.trainable=False
   
   model.compile(loss=loss, optimizer=Adam(lr=lr), metrics=[metric_f])
   return model, metric


def make_av_std(series, key, multitask=False, Noutputs=-1):
   if multitask:
      assert Noutputs>0
      arr = np.array([[x[key%i] for i in range(Noutputs)] for x in series]).mean(axis=1)
   else:
      arr = np.array([x[key] for x in series])
   n=np.sqrt(arr.shape[0]-1)
   if n>0: 
      std=arr.std(axis=0)/n
   else:
      std=np.zeros(arr.shape[1])
   return arr.mean(axis=0), std


class TimeDistributedMean(Layer):
   def build(self, input_shape):
      super(TimeDistributedMean, self).build(input_shape)
       
   # input shape (None, T, ...)
   # output shape (None, ...)
   def compute_output_shape(self, input_shape):
      return (input_shape[0],) + input_shape[2:]

   def call(self, x):
      return K.mean(x, axis=1)


class TimeDistributedMeanSquare(TimeDistributedMean):
   def call(self, x):
      return K.mean(x*x, axis=1)


def create_bayesian_uncertainity_model(model, epistemic_monte_carlo_simulations):
    inpt = Input(shape=(model.input_shape[1:]))
    x = RepeatVector(epistemic_monte_carlo_simulations)(inpt)
    # Keras TimeDistributed can only handle a single output from a model :(
    # and we technically only need the softmax outputs.
    x = TimeDistributed(model, name='epistemic_monte_carlo')(x)
    # predictive probabilities for each class
    softmax_mean = TimeDistributedMean(name='softmax_mean')(x)
    softmax_mean_square = TimeDistributedMeanSquare(name='softmax_mean_square')(x)
    monte_carlo_model = Model(inputs=inpt, outputs=[softmax_mean, softmax_mean_square])

    return monte_carlo_model


def make_bayesian_prediction(model, test_x, repetitions=1000):
   '''
   Bayesian prediction (from dropout distribution). 
   Assuming binary classiffication'''
   
   try:
      K.set_learning_phase(1)
      bayesian_model = create_bayesian_uncertainity_model(model, repetitions)
      mean_prob, mean_prob_sq = [arr[:,1] for arr in bayesian_model.predict(test_x)]
   except:
      mean_prob, mean_prob_sq = [], []
      #K.set_learning_phase(1)
      #for layer in model.layers:
      #   if 'ropout' in layer.name:
      #      print(dir(layer))
      #      layer.training=True
      #test_model = Model(inputs=model.inputs, outputs=model.outputs)
      #test_model.compile(loss='categorical_crossentropy', optimizer='adam')
      #K.set_learning_phase(1)
      #assert type(test_x).__name__=='list'
      if type(test_x).__name__=='tuple':
         test_x=list(test_x)
      #N=test_x[0].shape[0]
      #Nall=N*repetitions
      #repeated_indices = [idx%N for idx in range(Nall)]
      #repeated_data = [arr[repeated_indices] for arr in test_x]
      #mean_prob = model.predict(repeated_data)[:,1]
      #mean_prob = mean_prob.reshape(repetitions, N)
      #mean_prob_sq = mean_prob**2
      for _ in range(repetitions):
         #K.set_learning_phase(1)
         prob = model.predict(test_x)[:, 1]
         prob_sq = prob**2
         mean_prob.append(prob)
         mean_prob_sq.append(prob_sq)
      mean_prob = np.mean(mean_prob, axis=0)
      mean_prob_sq = np.mean(mean_prob_sq, axis=0)

   epistemic = mean_prob_sq - mean_prob**2
   aleatoric = mean_prob - mean_prob_sq
   total = aleatoric+epistemic
   
   assert (abs(mean_prob-mean_prob**2 - total) <0.00001).all()
   K.set_learning_phase(0)
   
   return {'mean_prob': mean_prob, 'epistemic':epistemic, 'aleatoric':aleatoric}


def make_ggnn2_model(input_shapes, output_shape, model_config):
   '''
   order:
   
   X_input, filters_input, nums_input, identity_input, adjacency_input 
   '''
   #training_data = ([X, graph_conv_filters, lens], Y)
   #X_input, filters_input, nums_input, identity_input, adjacency_input 
   features_shape, filters_shape, lens_shape, identity_shape, adjacency_shape = input_shapes
   filters_shape1, max_atoms = filters_shape[1:]
   
   X_input = Input(shape=features_shape[1:])
   filters_input = Input(shape=filters_shape[1:])
   identity_input = Input(shape=identity_shape[1:])
   adjacency_input = Input(shape=adjacency_shape[1:])
   nums_input= Input(shape=(None,))

   num_filters = int(filters_shape[1]/max_atoms)
   model_config['max_atoms'] = max_atoms
   model_config['num_filters'] = num_filters

   #control parameters
   N_H = model_config.get('hidden_units', 128)
   dropout_prob = model_config.get('dropout', 0.031849402173891934)
   lr = model_config.get('lr', 1e-3)
   l2_val = model_config.get('l2', 1e-3)
   N_it = model_config.get('num_layers', 8)
   activation = model_config.get('activation', 'softplus')
   drp_flag = model_config.get('dropout_flag', False)
   
   #initial convolution
   H = MultiGraphCNN(N_H, 1, activation=activation, kernel_regularizer=l2(l2_val), name='gcnn1')([X_input, identity_input])
   H = BatchNormalization()(H)
   H=Dropout(dropout_prob)(H, training=drp_flag)
   for it in range(N_it):
      H = MultiGraphCNN(N_H, num_filters, activation=activation, kernel_regularizer=l2(l2_val))([H, filters_input])
      H = Dropout(dropout_prob)(H, training=drp_flag)

   #Pooling
   output = Lambda(lambda X: K.sum(X[0], axis=1)/X[1])([H, nums_input])  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
   if len(output_shape)==2:
      N_output=2
      output_activation = 'softmax'
      loss_f='categorical_crossentropy'
      metric = 'categorical_accuracy'
   else:
      N_output=1
      output_activation = 'sigmoid'
      loss_f='binary_crossentropy'
      metric = 'accuracy'
   
   output = Dropout(dropout_prob)(output, training=drp_flag)
   output = Dense(N_output, activation=output_activation)(output)
   
   model = Model(inputs=[X_input, filters_input, nums_input, identity_input, adjacency_input ] , outputs=output)
   model.compile(loss=loss_f, optimizer=Adam(lr=lr), metrics=[metric])

   return model, metric


def make_ggnn_model(input_shapes, output_shape, model_config):
   '''
   order:
   
   X_input, filters_input, nums_input, identity_input, adjacency_input 
   '''
   #training_data = ([X, graph_conv_filters, lens], Y)
   #X_input, filters_input, nums_input, identity_input, adjacency_input 
   features_shape, filters_shape, lens_shape, identity_shape, adjacency_shape = input_shapes
   filters_shape1, max_atoms = filters_shape[1:]
   
   X_input = Input(shape=features_shape[1:])
   filters_input = Input(shape=filters_shape[1:])
   identity_input = Input(shape=identity_shape[1:])
   adjacency_input = Input(shape=adjacency_shape[1:])
   nums_input= Input(shape=(None,))

   num_filters = int(filters_shape[1]/max_atoms)
   model_config['max_atoms'] = max_atoms
   model_config['num_filters'] = num_filters

   #control parameters
   recurrent = model_config.get('recurrent', False)
   N_H = model_config.get('hidden_units', 128)
   dropout_prob = model_config.get('dropout', 0.031849402173891934)
   lr = model_config.get('lr', 1e-3)
   l2_val = model_config.get('l2', 1e-3)
   N_it = model_config.get('num_layers', 8)
   activation = model_config.get('activation', 'softplus')
   drp_flag = model_config.get('dropout_flag', False)
   
   #GGNN unit components
   hadamard = Lambda(lambda x: x[0]*x[1])
   sum_ = Lambda(lambda x: x[0]+x[1])
   tanh = Activation('tanh')
   combiner = Lambda(lambda x:(K.ones_like(x[0])-x[0])*x[1]+x[0]*x[2])
   def make_gated_unit_layer(N_H):
      GCN_Z = MultiGraphCNN(N_H, 2, activation='sigmoid', kernel_regularizer=l2(l2_val))
      GCN_R = MultiGraphCNN(N_H, 2, activation='sigmoid', kernel_regularizer=l2(l2_val))
      GCN_U = MultiGraphCNN(N_H, 1, activation='linear', use_bias=False, kernel_regularizer=l2(l2_val))
      GCN_W = MultiGraphCNN(N_H, 1, activation='linear', kernel_regularizer=l2(l2_val))
      return [GCN_Z,GCN_R,GCN_U,GCN_W]

   #initial convolution
   H = MultiGraphCNN(N_H, 1, activation=activation, kernel_regularizer=l2(l2_val), name='gcnn1')([X_input, identity_input])
   H = BatchNormalization()(H)
   H=Dropout(dropout_prob)(H, training=drp_flag)
   #GCNN iterations
   GCN_layers=[]
   GCN_layers.append(make_gated_unit_layer(N_H))
   for it in range(N_it):
      GCN_Z, GCN_R, GCN_U, GCN_W = GCN_layers[-1]
      z = GCN_Z([H, filters_input])
      r = GCN_R([H, filters_input])
      r = Dropout(dropout_prob)(r, training=drp_flag)
      z = Dropout(dropout_prob)(z, training=drp_flag)
      u = hadamard([r,H])
      u = GCN_U([u, identity_input])
      u = Dropout(dropout_prob)(u, training=drp_flag)
      w = GCN_W([H, adjacency_input])
      w = Dropout(dropout_prob)(w, training=drp_flag)
      HT = tanh(sum_([u,w]))
      H = combiner([z,H,HT]) 
      if it<(N_it-1) and not recurrent:
         GCN_layers.append(make_gated_unit_layer(N_H))

   #Pooling
   output = Lambda(lambda X: K.sum(X[0], axis=1)/X[1])([H, nums_input])  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
   if len(output_shape)==2:
      N_output=2
      output_activation = 'softmax'
      loss_f='categorical_crossentropy'
      metric = 'categorical_accuracy'
   else:
      N_output=1
      output_activation = 'sigmoid'
      loss_f='binary_crossentropy'
      metric = 'accuracy'
   
   output = Dropout(dropout_prob)(output, training=drp_flag)
   output = Dense(N_output, activation=output_activation)(output)
   
   model = Model(inputs=[X_input, filters_input, nums_input, identity_input, adjacency_input ] , outputs=output)
   model.compile(loss=loss_f, optimizer=Adam(lr=lr), metrics=[metric])

   return model, metric


def make_compressed_ggnn_model(data_sizes, output_shape, model_config):
   #'Nfeatures': Nfeatures, 'Natoms':Natoms, 'Nbonds':Nbonds 
   #features_shape, filters_shape, lens_shape, identity_shape, adjacency_shape
   #ASSUMING FIRST ORDER FILTER!
   Nfeatures = data_sizes['Nfeatures']
   Natoms = data_sizes['Natoms']
   Nbonds = data_sizes['Nbonds']
   pos_x = Nfeatures*Natoms
   pos_b = Nbonds*2 + pos_x
    
   features_shape = (-1, Natoms, Nfeatures)
   filters_shape = (-1, 2*Natoms, Natoms)
   lens_shape = (-1,)
   identity_shape = (-1, Natoms, Natoms)
   adjacency_shape = (-1, Natoms, Natoms)
   shapes = (features_shape, filters_shape, lens_shape, identity_shape, adjacency_shape)
    
   ggnn_model = make_ggnn_model(shapes, output_shape, model_config)[0]
    
   input_layer = Input(shape=(pos_b+1,))
   rcs_X = Lambda(lambda x:x[:,:pos_x])(input_layer)
   bond_indices = Lambda(lambda x: K.cast(x[:,pos_x:pos_b], 'int32'))(input_layer)
   lens = Lambda(lambda x:x[:,pos_b:])(input_layer)
    
   rcs_X = Reshape((Natoms, Nfeatures))(rcs_X)
   bond_indices = Reshape((Nbonds, 2))(bond_indices)
    
   def make_laplacian(bnd, Natoms): 
      bnd = K.cast(bnd, 'int32')
      prev = K.one_hot(bnd[:,:,0], Natoms)
      nxt = K.one_hot(bnd[:,:,1], Natoms)
      incidence = nxt-prev
            
      incidence_t = K.permute_dimensions(incidence, (0, 2, 1))
      lap = K.batch_dot(incidence_t, incidence)
      return lap
    
   def make_identity3d(ref3d):
      dg = K.ones_like(ref3d)[:,:,0]
      idnt = tf.linalg.diag(dg)
      return idnt
    
   def make_adj_from_laplacian(lap_and_identity):
       laplacian, identity = lap_and_identity
       mask = K.ones_like(laplacian) - identity
       return -laplacian*mask
 
       iden
    
   lap = Lambda(lambda x: make_laplacian(x, Natoms))(bond_indices)
   idnt = Lambda(make_identity3d)(lap)
   adj = Lambda(make_adj_from_laplacian)([lap, idnt])
   flt = Lambda(lambda x: K.concatenate(x, axis=1))([idnt,adj])
    
   transformed_stuff = [rcs_X, flt, lens, idnt, adj]
   print([xx.shape for xx in transformed_stuff])
   output = ggnn_model(transformed_stuff)
    
   if len(output_shape)==2:
       N_output=2
       output_activation = 'softmax'
       loss_f='categorical_crossentropy'
       metric = 'categorical_accuracy'
   else:
       N_output=1
       output_activation = 'sigmoid'
       loss_f='binary_crossentropy'
       metric = 'accuracy'
   lr = model_config.get('lr',1e-3) 
      
   model = Model(input=input_layer , outputs=output)
   model.compile(loss=loss_f, optimizer=Adam(lr=lr), metrics=[metric])
    
   return model, metric


def make_multitask_model(input_shape, output_shape, model_config):
   input_layer = Input(shape=(input_shape,))
   N_outputs=output_shape[0]
   if len(output_shape)==2:
      output_activation = 'sigmoid' 
      loss='binary_crossentropy'
      metric='accuracy'
      output_shape=1
   elif len(output_shape)==3:
      output_activation = 'softmax'
      loss = 'categorical_crossentropy'
      metric = 'categorical_accuracy'
      output_shape=2
   else:
      raise ValueError('output should be 2 or 3D, got %i'%len(output_shape))
    
   #get params
   balance_metric = model_config.get('metric_balance', False)
   if balance_metric:
      metric = 'balanced_'+metric
      metric_f = balanced_metrics[metric]
   else:
      metric = 'acc' if metric=='accuracy' else metric
      metric_f = metric

   drp = model_config.get('model_dropout', 0.2)
   drp_flag = model_config.get('dropout_flag', False)
   l2_val = model_config.get('model_l2', 1e-3)
   activation = model_config.get('activation', 'relu')
   init_weights_file = model_config.get('init_weights', None)
   lr = model_config.get('model_lr', 1e-3)

   n_hidden = model_config.get('hidden_units', 90)
   n_layers = model_config.get('num_common_layers', 1)
   if type(n_hidden).__name__ == 'int':
      n_hidden = [n_hidden for _ in range(n_layers)]
   #make model
   hidden = Dropout(drp)(input_layer, training=drp_flag)
   for N in n_hidden:
      hidden = Dense(N, activation=activation, 
                     kernel_regularizer=l2(l2_val))(hidden)
      hidden = Dropout(drp)(hidden, training=drp_flag)
   
   outputs=[]
   for i in range(N_outputs):
      output = Dense(output_shape, activation=output_activation, 
                  kernel_regularizer=l2(l2_val), name='out%i'%i)(hidden)
      outputs.append(output)
    
   model = Model(inputs=input_layer, outputs=outputs)

   #assign weights
   init_available = not( init_weights_file is None)
   if init_available:
      init_available = isfile(init_weights_file)
    
   if init_available:
      weights = gz_unpickle(init_weights_file)
      model.set_weights(weights)
   
   model.compile(loss=loss, optimizer=Adam(lr=lr), metrics=[metric_f])
   return model, metric


def make_model(input_shape, output_shape, model_config):
   kind = model_config.get('kind', 'rdkit_ae')
   if kind in ['rdkit_ae', 'mold2_ae']:
      return make_ae_model(input_shape, output_shape, model_config)
   elif kind=='multitask':
      return make_multitask_model(input_shape, output_shape, model_config)
   elif kind =='ggnn':
      return make_ggnn_model(input_shape, output_shape, model_config)
   elif kind=='ggnn_compressed':
      return make_compressed_ggnn_model(input_shape, output_shape, model_config)
   elif kind=='gcnn':
      return make_ggnn2_model(input_shape, output_shape, model_config)
   else:
      raise KeyError('Unknown model kind')
