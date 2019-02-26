import unittest

import numpy as np
from sympy import Matrix

from skrobot.symbol_math import round_matrix


class TestMath(unittest.TestCase):

    def test_round_matrix(self):
        assert(round_matrix(np.eye(4)) == Matrix(np.eye(4)))
