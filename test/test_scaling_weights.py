import unittest
import sys, os
curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.split(curr_dir)[0])
from data_preprocess import scale_weights, np

class TestScaling(unittest.TestCase):
   longMessage=True
   w = np.ones(9)
   s = 0.3
   y = np.array([  1,   1,   0,   0,   0,   1,   0,   0,   0])
   scaled_output = np.array([0.3, 0.3, 0.7, 0.7, 0.7, 0.3, 0.7, 0.7, 0.7])
   
  
   def test_non_w_modify(self):
      pre = hash(tuple(self.w))
      _ = scale_weights(self.y, self.w, self.s)
      after = hash(tuple(self.w))
      self.assertEqual(pre, after)

   def test_scaling_binary(self):
      expected_output = self.scaled_output
      actual_output = scale_weights(self.y, self.w, self.s)
      msg = '\nExpected : %s\nActual : %s'%(str(expected_output), str(actual_output))
      self.assertTrue((expected_output==actual_output).all(), msg)
      
      
   def test_scaling_categorical(self):
      y = np.array([ [0,1] if x else [1,0] for x in self.y])
      expected_output = self.scaled_output
      actual_output = scale_weights(y, self.w, self.s)
      msg = '\nExpected : %s\nActual : %s'%(str(expected_output), str(actual_output))
      self.assertTrue((expected_output==actual_output).all(), msg)
      

   def test_scaling_with_none_binary(self):
      expected_output = self.w
      actual_output = scale_weights(self.y, None, self.s)
      msg = '\nExpected : %s\nActual : %s'%(str(expected_output), str(actual_output))
      self.assertTrue((expected_output==actual_output).all(), msg)
      

   def test_scaling_with_none_categorical(self):
      y = np.array([ [0,1] if x else [1,0] for x in self.y])
      expected_output = self.w
      actual_output = scale_weights(y, None, self.s)
      msg = '\nExpected : %s\nActual : %s'%(str(expected_output), str(actual_output))
      self.assertTrue((expected_output==actual_output).all(), msg)


if __name__=='__main__':
   unittest.main()

