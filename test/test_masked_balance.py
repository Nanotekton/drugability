import unittest
import sys, os
curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.split(curr_dir)[0])
from data_preprocess import balance_masked_weights, np
from models import masked_balanced_score

class TestBalance(unittest.TestCase):
   longMessage=True
   known_values = {'y':np.array([  1,   1,   0,   0,   0,   1,   0,   0,   0]), 
                   'w':np.array([  1,   1,   0,   1,   0,   1,   0,   0,   1]), 
                   'o':np.array([0.4, 0.4, 0.0, 0.6, 0.0, 0.4, 0.0, 0.0, 0.6])}

   def test_detect_is_none(self):
      self.assertTrue(balance_masked_weights(self.known_values['y'], None) is None)


   def test_balance_masked_weights(self):
      resp = balance_masked_weights(self.known_values['y'], self.known_values['w'])
      msg = '\nGot: %s\nShould be: %s'%( str(resp), str(self.known_values['o']))

      self.assertTrue((resp==self.known_values['o']).all(), msg)


   def test_source_unchanged(self):
      prev = hash(tuple(self.known_values['w']))
      new = balance_masked_weights(self.known_values['y'], self.known_values['w'])
      after = hash(tuple(self.known_values['w']))
      self.assertEqual(prev, after)
   

   def test_one_class_empty(self):
      w = np.array([  1,   1,   0,   1])
      y = np.array([  1,   1,   0,   1])
      resp1 = balance_masked_weights(y, w)
      resp2 = balance_masked_weights(1-y, w)
      self.assertTrue((w==resp1).all())
      self.assertTrue((w==resp2).all())


class TestBalanceScore(unittest.TestCase):
   known_values = {'y_true':np.array([  1,   1,   0,   0,   0,   1,   0,   0,   0]),
                        'w':np.array([  1,   1,   0,   1,   0,   1,   0,   0,   1]),
                   'y_pred':np.array([  0,   1,   1,   0,   0,   1,   0,   0,   1]),
                     'bacc':7.0/12.0,
                      'acc':0.6} 

   def test_accuracy(self):
      calc_acc = masked_balanced_score(self.known_values['y_pred'], self.known_values['y_true'], self.known_values['w'], balance=False)
      self.assertAlmostEqual(calc_acc, self.known_values['acc'], delta=1e-9)

   
   def test_balanced_accuracy(self):
      calc_acc = masked_balanced_score(self.known_values['y_pred'], self.known_values['y_true'], self.known_values['w'], balance=True)
      self.assertAlmostEqual(calc_acc, self.known_values['bacc'], delta=1e-9)


if __name__=='__main__':
   unittest.main()

