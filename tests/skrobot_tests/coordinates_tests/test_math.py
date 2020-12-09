import math
import unittest

import numpy as np
from numpy import pi
from numpy import testing

from skrobot.coordinates.math import _check_valid_rotation
from skrobot.coordinates.math import angle_between_vectors
from skrobot.coordinates.math import cross_product
from skrobot.coordinates.math import matrix2quaternion
from skrobot.coordinates.math import matrix_exponent
from skrobot.coordinates.math import matrix_log
from skrobot.coordinates.math import midrot
from skrobot.coordinates.math import normalize_vector
from skrobot.coordinates.math import outer_product_matrix
from skrobot.coordinates.math import quaternion2matrix
from skrobot.coordinates.math import quaternion_conjugate
from skrobot.coordinates.math import quaternion_distance
from skrobot.coordinates.math import quaternion_from_axis_angle
from skrobot.coordinates.math import quaternion_inverse
from skrobot.coordinates.math import quaternion_multiply
from skrobot.coordinates.math import quaternion_norm
from skrobot.coordinates.math import quaternion_normalize
from skrobot.coordinates.math import quaternion_slerp
from skrobot.coordinates.math import random_quaternion
from skrobot.coordinates.math import random_rotation
from skrobot.coordinates.math import random_translation
from skrobot.coordinates.math import rodrigues
from skrobot.coordinates.math import rotate_matrix
from skrobot.coordinates.math import rotate_vector
from skrobot.coordinates.math import rotation_angle
from skrobot.coordinates.math import rotation_distance
from skrobot.coordinates.math import rotation_matrix
from skrobot.coordinates.math import rotation_matrix_from_axis
from skrobot.coordinates.math import rotation_matrix_from_rpy
from skrobot.coordinates.math import rotation_vector_to_quaternion
from skrobot.coordinates.math import rpy2quaternion
from skrobot.coordinates.math import rpy_angle
from skrobot.coordinates.math import rpy_matrix
from skrobot.coordinates.math import triple_product
from skrobot.coordinates.math import wxyz2xyzw
from skrobot.coordinates.math import xyzw2wxyz


class TestMath(unittest.TestCase):

    def test__check_valid_rotation(self):
        valid_rotation = np.eye(3)
        testing.assert_equal(
            _check_valid_rotation(valid_rotation),
            valid_rotation)

        invalid_rotation = np.arange(9).reshape(3, 3)
        with self.assertRaises(ValueError):
            _check_valid_rotation(invalid_rotation)

    def test_xyzw2wxyz(self):
        xyzw = np.array([0, 0, 0, 1])
        wxyz = xyzw2wxyz(xyzw)
        testing.assert_equal(
            wxyz, np.array([1, 0, 0, 0]))

    def test_wxyz2xyzw(self):
        wxyz = np.array([1, 0, 0, 0])
        xyzw = wxyz2xyzw(wxyz)
        testing.assert_equal(
            xyzw, np.array([0, 0, 0, 1]))

    def test_triple_product(self):
        a = np.array([1, 0, 3])
        b = np.array([2, 0, 0])
        c = np.array([0, 1, 0])
        ret = triple_product(a, b, c)
        self.assertEqual(ret, 6)

    def test_midrot(self):
        m1 = rotate_matrix(rotate_matrix(rotate_matrix(
            np.eye(3), 0.2, 'x'), 0.4, 'y'), 0.6, 'z')
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
                      [-0.587785, 0.700629, 0.404508]]),
            decimal=5)

        testing.assert_almost_equal(
            rpy_matrix(-pi, 0, pi / 2),
            np.array([[-1, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0]]))
        testing.assert_almost_equal(
            rpy_matrix(0, 0, 0),
            np.eye(3))

    def test_rpy_angle(self):
        a, b = rpy_angle(rpy_matrix(pi / 6, pi / 5, pi / 3))
        testing.assert_almost_equal(
            a, np.array([pi / 6, pi / 5, pi / 3]))
        testing.assert_almost_equal(
            b, np.array([3.66519143, 2.51327412, -2.0943951]))

        rot = np.array([[0, 0, 1],
                        [0, -1, 0],
                        [1, 0, 0]])
        testing.assert_almost_equal(
            rpy_angle(rot)[0],
            np.array([0, - pi / 2.0, pi]))

    def test_rotation_matrix(self):
        testing.assert_almost_equal(
            rotation_matrix(pi, [1, 1, 1]),
            np.array([[-0.33333333, 0.66666667, 0.66666667],
                      [0.66666667, -0.33333333, 0.66666667],
                      [0.66666667, 0.66666667, -0.33333333]]))

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

    def test_rodrigues(self):
        mat = rpy_matrix(pi / 6, pi / 5, pi / 3)
        theta, axis = rotation_angle(mat)
        rec_mat = rodrigues(axis, theta)
        testing.assert_array_almost_equal(mat, rec_mat)

        mat = rpy_matrix(- pi / 6, - pi / 5, - pi / 3)
        theta, axis = rotation_angle(mat)
        rec_mat = rodrigues(axis, theta)
        testing.assert_array_almost_equal(mat, rec_mat)

        # case of theta is None
        axis = np.array([np.pi, 0, 0], 'f')
        rec_mat = rodrigues(axis)
        testing.assert_array_almost_equal(
            rpy_angle(rec_mat)[0], np.array([0.0, 0.0, -np.pi], 'f'))

    def test_rotate_vector(self):
        testing.assert_array_almost_equal(
            rotate_vector([1, 0, 0], pi / 6.0, [1, 0, 0]),
            (1, 0, 0))
        testing.assert_array_almost_equal(
            rotate_vector([1, 0, 0], pi / 6.0, [0, 1, 0]),
            (0.8660254, 0, -0.5))
        testing.assert_array_almost_equal(
            rotate_vector([1, 0, 0], pi / 6.0, [0, 0, 1]),
            (0.8660254, 0.5, 0))

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

    def test_cross_product(self):
        testing.assert_array_equal(
            cross_product([-1, 2, -3], [1, 2, 3]),
            [12, 0, -4])

    def test_matrix_exponent(self):
        m1 = rotate_matrix(rotate_matrix(rotate_matrix(
            np.eye(3), 0.2, 'x'), 0.4, 'y'), 0.6, 'z')
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
            np.array([[1., 0., 0.],
                      [0., 0., -1.],
                      [0., 1., 0.]]))
        testing.assert_almost_equal(
            quaternion2matrix(
                normalize_vector([1.0,
                                  1 / np.sqrt(2),
                                  1 / np.sqrt(2),
                                  1 / np.sqrt(2)])),
            np.array([[0.2000000, -0.1656854, 0.9656854],
                      [0.9656854, 0.2000000, -0.1656854],
                      [-0.1656854, 0.9656854, 0.2000000]]))
        testing.assert_almost_equal(
            quaternion2matrix(
                normalize_vector([1.0,
                                  - 1 / np.sqrt(2),
                                  1 / np.sqrt(2),
                                  - 1 / np.sqrt(2)])),
            np.array([[0.2000000, 0.1656854, 0.9656854],
                      [-0.9656854, 0.2000000, 0.1656854],
                      [-0.1656854, -0.9656854, 0.2000000]]))
        testing.assert_almost_equal(
            quaternion2matrix([0.925754, 0.151891, 0.159933, 0.307131]),
            rotate_matrix(
                rotate_matrix(
                    rotate_matrix(
                        np.eye(3), 0.2, 'x'), 0.4, 'y'), 0.6, 'z'),
            decimal=5)

    def test_normalize_vector(self):
        testing.assert_almost_equal(
            normalize_vector([5, 0, 0]),
            np.array([1, 0, 0]))

        testing.assert_almost_equal(
            np.linalg.norm(normalize_vector([1, 1, 1])),
            1.0)

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
            matrix2quaternion(np.array([[0.428571, 0.514286, -0.742857],
                                        [-0.857143, -0.028571, -0.514286],
                                        [-0.285714, 0.857143, 0.428571]])),
            normalize_vector(np.array([4, 3, -1, -3])),
            decimal=5)

    def test_rotation_matrix_from_rpy(self):
        testing.assert_almost_equal(
            rotation_matrix_from_rpy([-pi, 0, pi / 2]),
            np.array([[-1, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0]]))
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

    def test_rotation_matrix_from_axis(self):
        x_axis = (1, 0, 0)
        y_axis = (0, 1, 0)
        rot = rotation_matrix_from_axis(x_axis, y_axis)
        _check_valid_rotation(rot)
        testing.assert_array_almost_equal(rot, np.eye(3))

        x_axis = (1, 1, 1)
        y_axis = (0, 0, 1)
        rot = rotation_matrix_from_axis(x_axis, y_axis)
        testing.assert_array_almost_equal(
            rot, [[0.57735027, -0.40824829, 0.70710678],
                  [0.57735027, -0.40824829, -0.70710678],
                  [0.57735027, 0.81649658, 0.0]])

        x_axis = (1, 1, 1)
        y_axis = (0, 0, -1)
        rot = rotation_matrix_from_axis(x_axis, y_axis)
        _check_valid_rotation(rot)
        testing.assert_array_almost_equal(
            rot, [[0.57735027, 0.40824829, -0.70710678],
                  [0.57735027, 0.40824829, 0.70710678],
                  [0.57735027, -0.81649658, 0.0]])

        rot = rotation_matrix_from_axis(y_axis, x_axis, axes='yx')
        _check_valid_rotation(rot)

    def test_rotation_distance(self):
        mat1 = np.eye(3)
        mat2 = np.eye(3)
        diff_theta = rotation_distance(mat1, mat2)
        self.assertEqual(diff_theta, 0.0)

        mat1 = rpy_matrix(0, 0, np.pi)
        mat2 = np.eye(3)
        diff_theta = rotation_distance(mat1, mat2)
        self.assertEqual(diff_theta, np.pi)

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

        # batch
        q0 = [[1, 0, 0, 0], [0, 1, 0, 0]]
        q1 = quaternion_conjugate(q0)
        q = quaternion_multiply(q0, q1)
        testing.assert_array_equal(
            q, [[1, 0, 0, 0], [1, 0, 0, 0]])

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

        # batch
        q0 = [[1, 0, 0, 0], [1, 2, 3, 4]]
        q1 = quaternion_inverse(q0)
        q = quaternion_multiply(q0, q1)
        testing.assert_almost_equal(
            q, [[1, 0, 0, 0], [1, 0, 0, 0]])

    def test_quaternion_slerp(self):
        q0 = [-0.84289035, -0.14618244, -0.12038416, 0.50366081]
        q1 = [0.28648105, -0.61500146, 0.73395791, 0.03174259]
        q = quaternion_slerp(q0, q1, 0.0)
        testing.assert_almost_equal(q, q0)

        q = quaternion_slerp(q0, q1, 1.0)
        testing.assert_almost_equal(q, q1)

        q = quaternion_slerp(q0, q1, 0.5)
        angle = math.acos(np.dot(q0, q))
        testing.assert_almost_equal(math.acos(-np.dot(q0, q1)) / angle,
                                    2.0)

    def test_quaternion_distance(self):
        q1 = rpy2quaternion([0, 0, 0])
        q2 = rpy2quaternion([0, 0, 0])
        self.assertEqual(quaternion_distance(q1, q2), 0.0)

        q1 = rpy2quaternion([np.pi, 0, 0])
        q2 = rpy2quaternion([0, 0, 0])
        self.assertEqual(quaternion_distance(q1, q2), np.pi)

        self.assertEqual(quaternion_distance(np.ones(4), np.ones(4)),
                         0.0)

    def test_quaternion_norm(self):
        q = np.array([1, 0, 0, 0])
        self.assertEqual(quaternion_norm(q), 1.0)

        q = np.array([0, 0, 0, 0])
        self.assertEqual(quaternion_norm(q), 0.0)

    def test_quaternion_normalize(self):
        q = np.array([1, 0, 0, 0])
        testing.assert_equal(
            quaternion_normalize(q),
            [1, 0, 0, 0])

        q = np.array([1, 2, 3, 4])
        testing.assert_almost_equal(
            quaternion_normalize(q),
            [0.18257419, 0.36514837, 0.54772256, 0.73029674])

    def test_quaternion_from_axis_angle(self):
        q = quaternion_from_axis_angle(0.1, [1, 0, 0])
        testing.assert_almost_equal(
            q,
            matrix2quaternion(rotation_matrix(0.1, [1, 0, 0])))

    def test_rotation_vector_to_quaternion(self):
        testing.assert_almost_equal(
            [1, 0, 0, 0],
            rotation_vector_to_quaternion([0, 0, 0]))

    def test_angle_between_vectors(self):
        v = (1., 1., 1.)
        theta = angle_between_vectors(v, v)
        testing.assert_almost_equal(theta, 0.0)

        unit_v = normalize_vector(v)
        theta = angle_between_vectors(unit_v, unit_v, normalize=False)
        testing.assert_almost_equal(theta, 0.0)

    def test_random_rotation(self):
        testing.assert_almost_equal(
            np.linalg.det(random_rotation()),
            1.0)

    def test_random_translation(self):
        random_translation()

    def test_random_quaternion(self):
        testing.assert_almost_equal(
            quaternion_norm(random_quaternion()),
            1.0)
