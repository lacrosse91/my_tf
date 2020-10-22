import numpy as np
import unittest
from dezero.core_simple import Variable



class VariableTest(unittest.TestCase):

    def test_sample(self):
        x = Variable(np.array(4.0))
        expected = np.array(4.0)
        self.assertEqual(x.data, expected)
