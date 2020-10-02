import numpy as np
from imblearn.keras import BalancedBatchGenerator
from keras.callbacks import Callback
from keras import backend as K

def balanced_metrics(y_true, y_pred, mode='bacc'):
   truep = K.cast(K.round(y_true), dtype='float32')
   truen = K.ones_like(truep)-truep
   predp = K.cast(K.round(y_pred), dtype='float32')
   predn = K.ones_like(predp)-predp
   num_truep = K.sum(truep)
   num_truen = K.sum(truen)
   true_negative = K.sum(truen*predn)
   true_positive = K.sum(truep*predp)
   tpr = true_positive/(num_truep+K.epsilon())
   tnr = true_negative/(num_truen+K.epsilon())
   bacc= 0.5*(tpr+tnr)
   stuff = {'tpr':tpr, 'tnr':tnr, 'bacc':bacc}
     
   return stuff[mode]


def balanced_categorical_metrics(y_true, y_pred, mode='bacc'):
   y_true2 = K.cast(K.argmax(y_true, axis=1), dtype='float32')
   y_pred2 = K.cast(K.argmax(y_pred, axis=1), dtype='float32')
   return balanced_metrics(y_true2, y_pred2, mode)


def balanced_categorical_accuracy(y_true, y_pred):
   return balanced_categorical_metrics(y_true, y_pred, 'bacc')


def balanced_accuracy(y_true, y_pred):
   return balanced_metrics(y_true, y_pred, 'bacc')


def categorical_tpr(y_true, y_pred):
   return balanced_categorical_metrics(y_true, y_pred, 'tpr')


def categorical_tnr(y_true, y_pred):
   return balanced_categorical_metrics(y_true, y_pred, 'tnr')


def tpr(y_true, y_pred):
   return balanced_metrics(y_true, y_pred, 'tpr')


def tnr(y_true, y_pred):
   return balanced_metrics(y_true, y_pred, 'tnr')


class BalancedAccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.tpr = []
        self.tnr = []

    def on_epoch_end(self, epoch, logs={}):
        x,y=self.validation_data[:2]
        y_p=self.model.predict(x)
        tpr, tnr=balanced_accuracy(y,y_p)
        bacc=0.5*(tpr+tnr)
        print('                                                                                  Validation balanced accuracy: %8.4f'%bacc)
        print('                                                                                                Validation tpr: %8.4f'%tpr)
        print('                                                                                                Validation tnr: %8.4f'%tnr)
        self.losses.append(bacc)
        self.tpr.append(tpr)
        self.tnr.append(tnr)
