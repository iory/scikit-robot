import unittest

import numpy as np
from sympy import Matrix

from skrobot.math import rotation_matrix
from skrobot.symbol_math import round_matrix


class TestMath(unittest.TestCase):

    def test_round_matrix(self):
        assert(round_matrix(np.eye(4)) == Matrix(np.eye(4)))

        rotation = np.eye(4)
        rotation[:3, :3] = rotation_matrix(np.pi / 2, [0, 0, 1])

        assert(round_matrix(rotation) == Matrix(
            [[0, -1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]))

        rotation = np.eye(4)
        rotation[:3, :3] = rotation_matrix(np.pi / 3, [1, 1, 1])
        round_matrix(rotation)
        assert(round_matrix(rotation) == Matrix([
            ['2/3', '-1/3', '2/3', 0],
            ['2/3', '2/3', '-1/3', 0],
            ['-1/3', '2/3', '2/3', 0],
            [0, 0, 0, 1]]))
