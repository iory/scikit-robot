import unittest

import numpy as np
from numpy import pi
from numpy import testing

from robot.math import matrix2quaternion
from robot.math import normalize_vector
from robot.math import quaternion2matrix
from robot.math import rotate_matrix
from robot.math import rotation_matrix_from_rpy
from robot.math import rpy_matrix


class TestMath(unittest.TestCase):

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
