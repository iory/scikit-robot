import math
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
from robot.math import quaternion_conjugate
from robot.math import quaternion_from_axis_angle
from robot.math import quaternion_inverse
from robot.math import quaternion_multiply
from robot.math import quaternion_slerp
from robot.math import rotate_matrix
from robot.math import rotation_angle
from robot.math import rotation_matrix
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

    def test_rpy_matrix(self):
        testing.assert_almost_equal(
            rpy_matrix(0, 0, 0),
            np.eye(3))

        testing.assert_almost_equal(
            rpy_matrix(pi / 6, pi / 5, pi / 3),
            np.array([[0.700629, 0.190839, 0.687531],
                      [0.404508, 0.687531, -0.603054],
                      [-0.587785, 0.700629, 0.404508]]))

    def test_rotation_matrix(self):
        testing.assert_almost_equal(
            rotation_matrix(pi, [1, 1, 1]),
            np.array([[-0.33333333,  0.66666667,  0.66666667],
                      [ 0.66666667, -0.33333333,  0.66666667],
                      [ 0.66666667,  0.66666667, -0.33333333]]))

        testing.assert_almost_equal(
            rotation_matrix(2 * pi, [1, 1, 1]),
            np.eye(3))

        testing.assert_almost_equal(
            rotation_matrix(pi / 3, [1, 0, 0]),
            np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.5, -0.866025],
                      [0.0, 0.866025, 0.5]]),
            decimal=5)

        testing.assert_almost_equal(
            rotation_matrix(pi / 3, [0, 1, 0]),
            np.array([[0.5, 0.0, 0.866025],
                      [0.0, 1.0, 0.0],
                      [-0.866025, 0.0, 0.5]]),
            decimal=5)

        testing.assert_almost_equal(
            rotation_matrix(pi / 3, [0, 0, 1]),
            np.array([[0.5, -0.866025, 0.0],
                      [0.866025, 0.5, 0.0],
                      [0.0, 0.0, 1.0]]),
            decimal=5)

    def test_rotation_angle(self):
        rot = rpy_matrix(-1.220e-08, -5.195e-09, 1.333e-09)
        with self.assertRaises(ValueError):
            rotation_angle(rot)

        rot = rpy_matrix(-1.220e-08, -5.195e-09, -1.333e-09)
        with self.assertRaises(ValueError):
            rotation_angle(rot)

        self.assertEqual(rotation_angle(np.eye(3)), None)

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

    def test_quaternion_multiply(self):
        q0 = [1, 0, 0, 0]
        q = quaternion_multiply(q0, q0)
        testing.assert_array_equal(
            q, [1, 0, 0, 0])

        q = quaternion_multiply([4, 1, -2, 3],
                                [8, -5, 6, 7])
        testing.assert_array_equal(
            q, [28, -44, -14, 48])

    def test_quaternion_conjugate(self):
        q0 = [1, 0, 0, 0]
        q1 = quaternion_conjugate(q0)
        q = quaternion_multiply(q0, q1)
        testing.assert_array_equal(
            q, [1, 0, 0, 0])

    def test_quaternion_inverse(self):
        q0 = [1, 0, 0, 0]
        q1 = quaternion_inverse(q0)
        q = quaternion_multiply(q0, q1)
        testing.assert_array_equal(
            q, [1, 0, 0, 0])

        q0 = [1, 2, 3, 4]
        q1 = quaternion_inverse(q0)
        q = quaternion_multiply(q0, q1)
        testing.assert_almost_equal(
            q, [1, 0, 0, 0])

    def test_quaternion_slerp(self):
        q0 = [-0.84289035, -0.14618244, -0.12038416,  0.50366081]
        q1 = [ 0.28648105, -0.61500146,  0.73395791,  0.03174259]
        q = quaternion_slerp(q0, q1, 0.0)
        testing.assert_almost_equal(q, q0)

        q = quaternion_slerp(q0, q1, 1.0)
        testing.assert_almost_equal(q, q1)

        q = quaternion_slerp(q0, q1, 0.5)
        angle = math.acos(np.dot(q0, q))
        testing.assert_almost_equal(math.acos(-np.dot(q0, q1)) / angle,
                                    2.0)

    def test_quaternion_from_axis_angle(self):
        q = quaternion_from_axis_angle(0.1, [1, 0, 0])
        testing.assert_almost_equal(
            q,
            matrix2quaternion(rotation_matrix(0.1, [1, 0, 0])))
