import unittest

import numpy as np
from numpy import pi
from numpy import testing

from robot.math import matrix2quaternion
from robot.math import matrix_exponent
from robot.math import matrix_log
from robot.math import midrot
from robot.math import normalize_vector
from robot.math import outer_product_matrix
from robot.math import quaternion2matrix
from robot.math import rotate_matrix
from robot.math import rotation_matrix_from_rpy
from robot.math import rpy_matrix


class TestMath(unittest.TestCase):

    def test_midrot(self):
        m1 = rotate_matrix(rotate_matrix(rotate_matrix(np.eye(3), 0.2, "x"), 0.4, "y"), 0.6, 'z')
        testing.assert_almost_equal(
            midrot(0.5, m1, np.eye(3)),
            np.array([[0.937735, -0.294516, 0.184158],
                      [0.319745, 0.939037, -0.126384],
                      [-0.135709, 0.177398, 0.974737]]),
            decimal=5)

    def test_outer_product_matrix(self):
        testing.assert_array_equal(outer_product_matrix([1, 2, 3]),
                                   np.array([[0.0, -3.0, 2.0],
                                             [3.0, 0.0, -1.0],
                                             [-2.0, 1.0, 0.0]]))

    def test_matrix_exponent(self):
        m1 = rotate_matrix(rotate_matrix(rotate_matrix(np.eye(3), 0.2, "x"), 0.4, "y"), 0.6, 'z')
        testing.assert_almost_equal(
            matrix_exponent(matrix_log(m1)), m1,
            decimal=5)

    def test_quaternion2matrix(self):
        testing.assert_array_equal(
            quaternion2matrix([1, 0, 0, 0]),
            np.eye(3))
        testing.assert_almost_equal(
            quaternion2matrix([1.0 / np.sqrt(2),
                               1.0 / np.sqrt(2),
                               0, 0]),
            np.array([[1., 0.,  0.],
                      [0., 0., -1.],
                      [0., 1.,  0.]]))
        testing.assert_almost_equal(
            quaternion2matrix(
                normalize_vector([1.0,
                                  1 / np.sqrt(2),
                                  1 / np.sqrt(2),
                                  1 / np.sqrt(2)])),
            np.array([[0.2000000, -0.1656854,  0.9656854],
                      [0.9656854,  0.2000000, -0.1656854],
                      [-0.1656854,  0.9656854,  0.2000000]]))
        testing.assert_almost_equal(
            quaternion2matrix(
                normalize_vector([1.0,
                                  - 1 / np.sqrt(2),
                                  1 / np.sqrt(2),
                                  - 1 / np.sqrt(2)])),
            np.array([[0.2000000,  0.1656854,  0.9656854],
                      [-0.9656854,  0.2000000,  0.1656854],
                      [-0.1656854, -0.9656854,  0.2000000]]))
        testing.assert_almost_equal(
            quaternion2matrix([0.925754, 0.151891, 0.159933, 0.307131]),
            rotate_matrix(
                rotate_matrix(
                    rotate_matrix(
                        np.eye(3), 0.2, 'x'), 0.4, 'y'), 0.6, 'z'),
            decimal=5)

    def test_matrix2quaternion(self):
        testing.assert_almost_equal(matrix2quaternion(np.eye(3)),
                                    np.array([1, 0, 0, 0]))

        m = rotate_matrix(
            rotate_matrix(
                rotate_matrix(
                    np.eye(3), 0.2, 'x'), 0.4, 'y'), 0.6, 'z')

        testing.assert_almost_equal(
            quaternion2matrix(matrix2quaternion(m)),
            m)

        testing.assert_almost_equal(
            matrix2quaternion(np.array([[0.428571,   0.514286, -0.742857],
                                        [-0.857143, -0.028571, -0.514286],
                                        [-0.285714,  0.857143,  0.428571]])),
            normalize_vector(np.array([4, 3, -1, -3])),
            decimal=5)

    def test_rpy_matrix(self):
        testing.assert_almost_equal(
            rpy_matrix(-pi, 0, pi / 2),
            np.array([[-1, 0, 0],
                      [ 0, 0, 1],
                      [ 0, 1, 0]]))
        testing.assert_almost_equal(
            rpy_matrix(0, 0, 0),
            np.eye(3))

    def test_rotation_matrix_from_rpy(self):
        testing.assert_almost_equal(
            rotation_matrix_from_rpy([-pi, 0, pi / 2]),
            np.array([[-1, 0, 0],
                      [ 0, 0, 1],
                      [ 0, 1, 0]]))
        testing.assert_almost_equal(
            rotation_matrix_from_rpy([0, 0, 0]),
            np.eye(3))

        # rotation_matrix_from_rpy and rpy_matrix should be same.
        for i in range(100):
            r = np.random.random()
            p = np.random.random()
            y = np.random.random()
            testing.assert_almost_equal(rpy_matrix(y, p, r),
                                        rotation_matrix_from_rpy([y, p, r]))
