import math
import sys
import unittest

import numpy as np
from numpy import pi
from numpy import testing
import pytest

from skrobot.coordinates.math import _check_valid_rotation
from skrobot.coordinates.math import angle_between_vectors
from skrobot.coordinates.math import axis_angle_vector_to_rotation_matrix
from skrobot.coordinates.math import clockwise_angle_between_vectors
from skrobot.coordinates.math import counter_clockwise_angle_between_vectors
from skrobot.coordinates.math import cross_product
from skrobot.coordinates.math import interpolate_rotation_matrices
from skrobot.coordinates.math import invert_yaw_pitch_roll
from skrobot.coordinates.math import matrix2quaternion
from skrobot.coordinates.math import matrix2rpy
from skrobot.coordinates.math import matrix2xyzrpy
from skrobot.coordinates.math import matrix2ypr
from skrobot.coordinates.math import matrix_relative
from skrobot.coordinates.math import normalize_vector
from skrobot.coordinates.math import quaternion2matrix
from skrobot.coordinates.math import quaternion2rpy
from skrobot.coordinates.math import quaternion_absolute_distance
from skrobot.coordinates.math import quaternion_conjugate
from skrobot.coordinates.math import quaternion_distance
from skrobot.coordinates.math import quaternion_from_axis_angle
from skrobot.coordinates.math import quaternion_from_vectors
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
from skrobot.coordinates.math import rotation_matrix_from_vectors
from skrobot.coordinates.math import rotation_matrix_to_axis_angle_vector
from skrobot.coordinates.math import rotation_vector_to_quaternion
from skrobot.coordinates.math import rpy2homogeneous
from skrobot.coordinates.math import rpy2matrix
from skrobot.coordinates.math import rpy2quaternion
from skrobot.coordinates.math import rpy_matrix
from skrobot.coordinates.math import skew_symmetric_matrix
from skrobot.coordinates.math import triple_product
from skrobot.coordinates.math import wxyz2xyzw
from skrobot.coordinates.math import xyzrpy2matrix
from skrobot.coordinates.math import xyzw2wxyz
from skrobot.coordinates.math import ypr2matrix


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

        # for batch
        xyzw = np.array([[0, 0, 0, 1],
                         [0, 0, 1, 0]])
        wxyz = xyzw2wxyz(xyzw)
        testing.assert_equal(
            wxyz, np.array([[1, 0, 0, 0],
                            [0, 0, 0, 1]]))

    def test_wxyz2xyzw(self):
        wxyz = np.array([1, 0, 0, 0])
        xyzw = wxyz2xyzw(wxyz)
        testing.assert_equal(
            xyzw, np.array([0, 0, 0, 1]))

        # for batch
        wxyz = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        xyzw = wxyz2xyzw(wxyz)
        testing.assert_equal(
            xyzw, np.array([[0, 0, 0, 1],
                            [1, 0, 0, 0]]))

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
            interpolate_rotation_matrices(0.5, m1, np.eye(3)),
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
        a, b = matrix2ypr(rpy_matrix(pi / 6, pi / 5, pi / 3)), np.array([3.66519143, 2.51327412, -2.0943951])
        testing.assert_almost_equal(
            a, np.array([pi / 6, pi / 5, pi / 3]))
        testing.assert_almost_equal(
            b, np.array([3.66519143, 2.51327412, -2.0943951]))

        rot = np.array([[0, 0, 1],
                        [0, -1, 0],
                        [1, 0, 0]])
        testing.assert_almost_equal(
            matrix2ypr(rot),
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
            matrix2ypr(rec_mat), np.array([0.0, 0.0, -np.pi], 'f'))

    def test_rodrigues_batch(self):
        # Creating batch matrices and angles
        angles = [pi / 6, -pi / 6, pi / 5, -pi / 5]
        axes = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]
        mats = [rpy_matrix(angle, angle, angle) for angle in angles]
        thetas, axis_list = zip(*(rotation_angle(mat) for mat in mats))

        rec_mats = rodrigues(axis_list, thetas)
        testing.assert_array_almost_equal(mats, rec_mats)

        # Test case where theta is None
        axes = np.array([[pi, 0, 0], [0, pi, 0]], dtype='float32')
        rec_mats = rodrigues(axes)

        # Expected rotations are around x and y axis by pi respectively
        expected_mats = [
            rpy_matrix(0, 0, pi),  # Rotation by pi around x-axis
            rpy_matrix(0, pi, 0)   # Rotation by pi around y-axis
        ]

        # Check that reconstructed matrices match expected results
        for rec_mat, expected_mat in zip(rec_mats, expected_mats):
            testing.assert_array_almost_equal(rec_mat, expected_mat)

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

        # Coordinates().rotate(np.pi / 2.0, 'y').rotation
        rot = [[0, 0, 1],
               [0, 1, 0],
               [-1, 0, 0]]
        theta, axis = rotation_angle(rot)
        testing.assert_array_almost_equal(theta, np.pi / 2.0)
        testing.assert_array_almost_equal(axis, [0, 1, 0])
        angular_velocity_vector = rotation_angle(
            rot, return_angular_velocity=True)
        testing.assert_array_almost_equal(
            angular_velocity_vector, [0, np.pi / 2.0, 0])

        self.assertEqual(rotation_angle(np.eye(3)), (None, None))
        self.assertEqual(
            rotation_angle(np.eye(3), return_angular_velocity=True),
            None)

    def test_outer_product_matrix(self):
        testing.assert_array_equal(skew_symmetric_matrix([1, 2, 3]),
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
            axis_angle_vector_to_rotation_matrix(rotation_matrix_to_axis_angle_vector(m1)), m1,
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

    def test_quaternion2matrix_batch(self):
        # Batched (N, 4) input must match the scalar path applied per element.
        qs = np.array([
            [1, 0, 0, 0],
            [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0, 0],
            normalize_vector([1.0, 0.3, -0.5, 0.2]),
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        mats = quaternion2matrix(qs)
        self.assertEqual(mats.shape, (len(qs), 3, 3))
        for i in range(len(qs)):
            _check_valid_rotation(mats[i])
            testing.assert_almost_equal(mats[i], quaternion2matrix(qs[i]))
        # Absolute check on the batched arithmetic for the x/y/z 180 deg cases.
        testing.assert_almost_equal(
            quaternion2matrix(np.array([[0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype=np.float64)),
            np.array([np.diag([1., -1., -1.]),
                      np.diag([-1., 1., -1.]),
                      np.diag([-1., -1., 1.])]))

    def test_quaternion2matrix_normalize_and_validation(self):
        # Default (normalize=False) rejects a non-unit quaternion, on both the
        # scalar (ndim == 1) and the batched (ndim == 2) validation paths.
        with self.assertRaises(ValueError):
            quaternion2matrix([1, 1, 0, 0])
        with self.assertRaises(ValueError):
            quaternion2matrix(np.array([[1, 1, 0, 0],
                                        [1, 0, 0, 0]], dtype=np.float64))
        # normalize=True normalizes the input first instead of raising.
        testing.assert_almost_equal(
            quaternion2matrix([1, 1, 0, 0], normalize=True),
            quaternion2matrix(normalize_vector([1, 1, 0, 0])))
        # A unit quaternion within tolerance is accepted unchanged.
        testing.assert_almost_equal(
            quaternion2matrix([1, 0, 0, 0]), np.eye(3))

    @staticmethod
    def _diverse_rotations():
        """A deterministic, non-degenerate set of rotation matrices.

        Covers all four ``matrix2quaternion`` trace branches with non-zero
        off-diagonal terms, both signs of every ``argmax`` selection, and the
        ``tr == 0`` boundary (120 deg rotations), so the branch conditions and
        per-term signs are all pinned down.  Uses a private ``RandomState`` so
        it never depends on (or perturbs) the global numpy RNG.
        """
        rs = np.random.RandomState(42)
        mats = []
        for _ in range(150):
            axis = rs.randn(3)
            axis = axis / np.linalg.norm(axis)
            mats.append(rotation_matrix(rs.uniform(-pi, pi), axis))
        for a in ([1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 1, 1], [1, -1, 2], [2, 1, 1]):
            mats.append(rotation_matrix(2 * pi / 3, a))   # tr == 0
            mats.append(rotation_matrix(-2 * pi / 3, a))
        return np.array(mats)

    def test_matrix2quaternion_branches(self):
        # matrix2quaternion selects one of four arithmetic branches from the
        # trace/diagonal; exercise each with a matrix that lands in it.
        cases = [
            (np.eye(3), [1, 0, 0, 0]),                       # tr > 0
            (rotate_matrix(np.eye(3), pi, 'x'), [0, 1, 0, 0]),  # m00 largest
            (rotate_matrix(np.eye(3), pi, 'y'), [0, 0, 1, 0]),  # m11 largest
            (rotate_matrix(np.eye(3), pi, 'z'), [0, 0, 0, 1]),  # else (m22)
        ]
        for m, q in cases:
            testing.assert_almost_equal(matrix2quaternion(m), q)
            testing.assert_almost_equal(
                quaternion2matrix(matrix2quaternion(m)), m)

        # The 180 deg cases above are diagonal, so the off-diagonal terms of
        # each branch stay zero and their signs are not pinned down.  General
        # rotations that land in every branch with non-zero off-diagonals must
        # round-trip exactly through the scalar path.
        for m in self._diverse_rotations():
            testing.assert_almost_equal(
                quaternion2matrix(matrix2quaternion(m)), m)

    def test_matrix2quaternion_batch(self):
        # Batched (N, 3, 3) input uses a separate vectorized code path.
        mats = np.array([
            np.eye(3),
            rotate_matrix(np.eye(3), pi, 'x'),
            rotate_matrix(np.eye(3), pi, 'y'),
            rotate_matrix(np.eye(3), pi, 'z'),
            rotate_matrix(rotate_matrix(rotate_matrix(
                np.eye(3), 0.2, 'x'), 0.4, 'y'), 0.6, 'z'),
        ])
        quats = matrix2quaternion(mats)
        self.assertEqual(quats.shape, (len(mats), 4))
        testing.assert_almost_equal(
            quats,
            np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0.925754, 0.151891, 0.159933, 0.307131]]),
            decimal=5)
        for i in range(len(mats)):
            testing.assert_almost_equal(
                quaternion2matrix(quats[i]), mats[i])

        # The vectorized path selects a per-element sign/branch from the
        # off-diagonal terms; diagonal-heavy matrices (identity, 180 deg)
        # leave those terms zero and cannot exercise it.  Use a deterministic
        # set of general rotations that collectively hit all four branches
        # and cross-check the batch against the (separately tested) scalar
        # path, up to the global sign ambiguity of quaternions.
        general = self._diverse_rotations()
        batch = matrix2quaternion(general)
        self.assertEqual(batch.shape, (len(general), 4))
        for i in range(len(general)):
            scalar = matrix2quaternion(general[i])
            self.assertTrue(
                np.allclose(batch[i], scalar, atol=1e-7)
                or np.allclose(batch[i], -scalar, atol=1e-7))
            testing.assert_almost_equal(
                quaternion2matrix(batch[i]), general[i])

        # Unsupported input shapes raise a clear error.
        with self.assertRaises(ValueError):
            matrix2quaternion(np.zeros(3))
        with self.assertRaises(ValueError):
            matrix2quaternion(np.zeros((2, 3, 3, 3)))

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

        with self.assertRaises(ValueError):
            rotation_matrix_from_axis((1, 0, 0), (-2, 0, 0))
        with self.assertRaises(ValueError):
            rotation_matrix_from_axis((1, 0, 0), (1, 0, 0))

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

        # batch
        testing.assert_equal(
            quaternion_distance(np.ones((10, 4)),
                                np.ones((10, 4))), np.zeros(10))

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

    def test_quaternion2rpy(self):
        # Test with a known simple quaternion representing no rotation
        q = np.array([1, 0, 0, 0])  # Identity quaternion
        expected_rpy = np.array([0, 0, 0])  # No rotation expected
        rpy, _ = quaternion2rpy(q)
        testing.assert_almost_equal(rpy, expected_rpy, decimal=5)

        # Test with a quaternion representing a 180 degree
        # rotation around the z-axis
        # Quaternion for 180 degree rotation around z
        q = np.array([0, 0, 0, 1])
        # 180 degrees yaw, no pitch or roll
        expected_rpy = np.array([np.pi, 0, 0])
        rpy, _ = quaternion2rpy(q)
        testing.assert_almost_equal(rpy, expected_rpy, decimal=5)

        # Test with an array of quaternions
        qs = np.array([
            [1, 0, 0, 0],  # No rotation
            [0, 1, 0, 0],  # 180 degree rotation around x
            [0, 0, 1, 0]   # 180 degree rotation around y
        ])
        expected_rpys = np.array([
            [0, 0, 0],
            [0, 0, np.pi],
            [np.pi, 0, np.pi],
        ])
        rpys, _ = quaternion2rpy(qs)
        testing.assert_array_almost_equal(rpys, expected_rpys, decimal=5)

        # Normalized quaternion edge case, 90 degrees rotation about x
        q = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
        expected_rpy = np.array([0, 0, np.pi / 2])  # 90 degrees pitch
        rpy, _ = quaternion2rpy(q)
        testing.assert_almost_equal(rpy, expected_rpy, decimal=5)

    def test_quaternion2rpy_batch_matches_scalar(self):
        # The batched path must agree with the scalar one.  Pitch has to be
        # non-zero here: at pitch 0 the two disagreeing formulas coincide.
        qs = np.array([
            rpy2quaternion([0.0, np.deg2rad(30), 0.0]),
            rpy2quaternion([0.4, -0.9, 0.2]),
            rpy2quaternion([-1.2, 0.7, 2.0]),
        ])
        expected = np.array([[0.0, np.deg2rad(30), 0.0],
                             [0.4, -0.9, 0.2],
                             [-1.2, 0.7, 2.0]])
        batch, batch_second = quaternion2rpy(qs)
        testing.assert_almost_equal(batch, expected)
        for i, q in enumerate(qs):
            scalar, second = quaternion2rpy(q)
            testing.assert_almost_equal(scalar, expected[i])
            testing.assert_almost_equal(batch[i], scalar)
            testing.assert_almost_equal(batch_second[i], second)

    def test_quaternion2rpy_at_gimbal_lock(self):
        # pitch = +/- 90 deg makes the angles ambiguous but not the rotation,
        # so the rebuilt matrix must still match.
        for pitch in (np.pi / 2, -np.pi / 2):
            m = rotation_matrix_from_rpy([0.0, pitch, 0.3])
            rpy, _ = quaternion2rpy(matrix2quaternion(m))
            testing.assert_almost_equal(
                rpy_matrix(rpy[0], rpy[1], rpy[2]), m)

    def test_quaternion_absolute_distance(self):
        q = normalize_vector([0.3, -0.5, 0.7, 0.2])
        # q and -q are the same rotation.
        testing.assert_almost_equal(quaternion_absolute_distance(q, -q), 0.0)
        testing.assert_almost_equal(quaternion_absolute_distance(q, q), 0.0)
        testing.assert_almost_equal(
            quaternion_absolute_distance(
                [1, 0, 0, 0], [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
            np.pi)
        # Never exceeds pi, unlike the sign-sensitive variant.
        self.assertLessEqual(
            quaternion_absolute_distance([1, 0, 0, 0], [-1, 0, 0, 0]), np.pi)

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

        # directed
        theta = angle_between_vectors(
            [1, 0, 0], [-1, 0, 0], directed=False)
        testing.assert_almost_equal(theta, 0.0)

        theta = angle_between_vectors(
            [1, 0, 0], [-1, 0, 0], directed=True)
        testing.assert_almost_equal(theta, np.pi)

    def test_counter_clockwise_angle_between_vectors(self):
        v2_and_angles = [([1, 0, 0], 0),
                         ([0.5, 0.5, 0], np.pi / 4),
                         ([0, 1, 0], np.pi / 2),
                         ([-0.5, 0.5, 0], np.pi * 3 / 4),
                         ([-1, 0, 0], np.pi),
                         ([-0.5, -0.5, 0], np.pi * 5 / 4),
                         ([0, -1, 0], np.pi * 3 / 2),
                         ([0.5, -0.5, 0], np.pi * 7 / 4)]
        normal_vector = [0, 0, 1]
        v1 = [1, 0, 0]
        for v2, valid_angle in v2_and_angles:
            angle = counter_clockwise_angle_between_vectors(
                v1, v2, normal_vector)
            testing.assert_equal(angle, valid_angle)
        v2_and_angles = [([1, 0, 0], 0),
                         ([0.5, 0.5, 0], 2 * np.pi - np.pi / 4),
                         ([0, 1, 0], 2 * np.pi - np.pi / 2),
                         ([-0.5, 0.5, 0], 2 * np.pi - np.pi * 3 / 4),
                         ([-1, 0, 0], 2 * np.pi - np.pi),
                         ([-0.5, -0.5, 0], 2 * np.pi - np.pi * 5 / 4),
                         ([0, -1, 0], 2 * np.pi - np.pi * 3 / 2),
                         ([0.5, -0.5, 0], 2 * np.pi - np.pi * 7 / 4)]
        v1 = [1, 0, 0]
        normal_vector = [0, 0, -1]
        for v2, valid_angle in v2_and_angles:
            angle = counter_clockwise_angle_between_vectors(
                v1, v2, normal_vector)
            testing.assert_equal(angle, valid_angle)

    def test_clockwise_angle_between_vectors(self):
        v2_and_angles = [([1, 0, 0], 0),
                         ([0.5, 0.5, 0], 2 * np.pi - np.pi / 4),
                         ([0, 1, 0], 2 * np.pi - np.pi / 2),
                         ([-0.5, 0.5, 0], 2 * np.pi - np.pi * 3 / 4),
                         ([-1, 0, 0], 2 * np.pi - np.pi),
                         ([-0.5, -0.5, 0], 2 * np.pi - np.pi * 5 / 4),
                         ([0, -1, 0], 2 * np.pi - np.pi * 3 / 2),
                         ([0.5, -0.5, 0], 2 * np.pi - np.pi * 7 / 4)]
        normal_vector = [0, 0, 1]
        v1 = [1, 0, 0]
        for v2, valid_angle in v2_and_angles:
            angle = clockwise_angle_between_vectors(
                v1, v2, normal_vector)
            testing.assert_equal(angle, valid_angle)
        v2_and_angles = [([1, 0, 0], 0),
                         ([0.5, 0.5, 0], np.pi / 4),
                         ([0, 1, 0], np.pi / 2),
                         ([-0.5, 0.5, 0], np.pi * 3 / 4),
                         ([-1, 0, 0], np.pi),
                         ([-0.5, -0.5, 0], np.pi * 5 / 4),
                         ([0, -1, 0], np.pi * 3 / 2),
                         ([0.5, -0.5, 0], np.pi * 7 / 4)]
        v1 = [1, 0, 0]
        normal_vector = [0, 0, -1]
        for v2, valid_angle in v2_and_angles:
            angle = clockwise_angle_between_vectors(
                v1, v2, normal_vector)
            testing.assert_equal(angle, valid_angle)

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

    @pytest.mark.skipif(sys.version_info[0] == 2, reason="Skip in Python 2")
    def test_invert_yaw_pitch_roll(self):

        # Test cases covering various scenarios
        test_cases = [
            # Basic cases
            ("Normal Case", (np.deg2rad(30), np.deg2rad(45), np.deg2rad(60))),
            ("Gimbal Lock at +90 deg", (np.deg2rad(45), np.deg2rad(90), np.deg2rad(30))),
            ("Gimbal Lock at -90 deg", (np.deg2rad(20), np.deg2rad(-90), np.deg2rad(50))),
            ("Zero Rotation", (0, 0, 0)),
            ("Yaw only", (np.deg2rad(90), 0, 0)),
            ("Pitch only", (0, np.deg2rad(-30), 0)),
            ("Roll only", (0, 0, np.deg2rad(120))),
            ("Negative angles", (np.deg2rad(-25), np.deg2rad(-50), np.deg2rad(-75))),

            # Boundary value cases
            ("Near gimbal lock +89.99", (np.deg2rad(30), np.deg2rad(89.99), np.deg2rad(45))),
            ("Near gimbal lock -89.99", (np.deg2rad(30), np.deg2rad(-89.99), np.deg2rad(45))),
            ("All 180 degrees", (np.deg2rad(180), np.deg2rad(180), np.deg2rad(180))),
            ("Small angles", (np.deg2rad(0.01), np.deg2rad(0.01), np.deg2rad(0.01))),
            ("Very small angles", (1e-6, 1e-6, 1e-6)),
            ("Alternating signs", (np.deg2rad(45), np.deg2rad(-45), np.deg2rad(45))),
            ("Edge case 1", (0, np.deg2rad(90), 0)),
            ("Edge case 2", (0, np.deg2rad(-90), 0)),

            # Precision cases
            ("Very close to +90 (1e-10)", (0, np.pi / 2 - 1e-10, 0)),
            ("Very close to -90 (1e-10)", (0, -np.pi / 2 + 1e-10, 0)),
            ("Near +90 with yaw/roll", (np.deg2rad(45), np.deg2rad(89.9999), np.deg2rad(30))),
            ("Near -90 with yaw/roll", (np.deg2rad(45), np.deg2rad(-89.9999), np.deg2rad(30))),

            # Extreme corner cases
            ("Multiple 90 degrees", (np.deg2rad(90), np.deg2rad(90), np.deg2rad(90))),
            ("All negative 90s", (np.deg2rad(-90), np.deg2rad(-90), np.deg2rad(-90))),
            ("Mixed extreme signs", (np.deg2rad(90), np.deg2rad(-90), np.deg2rad(90))),
            ("Symmetric case", (np.deg2rad(60), np.deg2rad(60), np.deg2rad(60))),
            ("Anti-symmetric", (np.deg2rad(60), np.deg2rad(-60), np.deg2rad(60))),
        ]

        tolerance = 1e-9

        for name, angles in test_cases:
            yaw, pitch, roll = angles

            # 1. Calculate original rotation matrix
            R_original = rpy_matrix(yaw, pitch, roll)

            # 2. Calculate inverse YPR using our function
            inv_yaw, inv_pitch, inv_roll = invert_yaw_pitch_roll(yaw, pitch, roll)

            # 3. Calculate inverse rotation matrix from inverse YPR
            R_inverted = rpy_matrix(inv_yaw, inv_pitch, inv_roll)

            # 4. Compose original and inverse rotations
            R_combined = np.matmul(R_original, R_inverted)

            # 5. Result should be identity matrix
            identity_matrix = np.identity(3)

            # Test that the combined rotation is identity
            testing.assert_allclose(
                R_combined, identity_matrix, atol=tolerance,
                err_msg="Failed for test case: {}".format(name)
            )

        # Additional specific tests for gimbal lock cases
        # Test exact gimbal lock at +90 degrees
        yaw, pitch, roll = np.deg2rad(45), np.pi / 2, np.deg2rad(30)
        inv_yaw, inv_pitch, inv_roll = invert_yaw_pitch_roll(yaw, pitch, roll)

        R_original = rpy_matrix(yaw, pitch, roll)
        R_inverted = rpy_matrix(inv_yaw, inv_pitch, inv_roll)
        R_combined = np.matmul(R_original, R_inverted)

        testing.assert_allclose(
            R_combined, np.identity(3), atol=tolerance,
            err_msg="Failed for exact +90 degree gimbal lock case"
        )

        # Test exact gimbal lock at -90 degrees
        yaw, pitch, roll = np.deg2rad(45), -np.pi / 2, np.deg2rad(30)
        inv_yaw, inv_pitch, inv_roll = invert_yaw_pitch_roll(yaw, pitch, roll)

        R_original = rpy_matrix(yaw, pitch, roll)
        R_inverted = rpy_matrix(inv_yaw, inv_pitch, inv_roll)
        R_combined = np.matmul(R_original, R_inverted)

        testing.assert_allclose(
            R_combined, np.identity(3), atol=tolerance,
            err_msg="Failed for exact -90 degree gimbal lock case"
        )

    def test_matrix2ypr(self):
        # Test identity matrix
        identity = np.eye(3)
        ypr = matrix2ypr(identity)
        testing.assert_almost_equal(ypr, np.array([0, 0, 0]))

        # Test known rotation: 90 degrees around Z (yaw)
        rot_z = np.array([[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]])
        ypr = matrix2ypr(rot_z)
        testing.assert_almost_equal(ypr, np.array([pi / 2, 0, 0]))

        # Test known rotation: 90 degrees around Y (pitch)
        rot_y = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [-1, 0, 0]])
        ypr = matrix2ypr(rot_y)
        testing.assert_almost_equal(ypr, np.array([0, pi / 2, 0]))

        # Test known rotation: 90 degrees around X (roll)
        rot_x = np.array([[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]])
        ypr = matrix2ypr(rot_x)
        testing.assert_almost_equal(ypr, np.array([0, 0, pi / 2]))

        # Test composite rotation and round-trip conversion
        yaw, pitch, roll = pi / 6, pi / 4, pi / 3
        rot = ypr2matrix(yaw, pitch, roll)
        recovered_ypr = matrix2ypr(rot)
        testing.assert_almost_equal(recovered_ypr, np.array([yaw, pitch, roll]))

    def test_matrix2rpy(self):
        # Test identity matrix
        identity = np.eye(3)
        rpy = matrix2rpy(identity)
        testing.assert_almost_equal(rpy, np.array([0, 0, 0]))

        # Test known rotation: 90 degrees around Z (yaw)
        rot_z = np.array([[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]])
        rpy = matrix2rpy(rot_z)
        testing.assert_almost_equal(rpy, np.array([0, 0, pi / 2]))

        # Test known rotation: 90 degrees around Y (pitch)
        rot_y = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [-1, 0, 0]])
        rpy = matrix2rpy(rot_y)
        testing.assert_almost_equal(rpy, np.array([0, pi / 2, 0]))

        # Test known rotation: 90 degrees around X (roll)
        rot_x = np.array([[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]])
        rpy = matrix2rpy(rot_x)
        testing.assert_almost_equal(rpy, np.array([pi / 2, 0, 0]))

        # Test composite rotation and round-trip conversion
        roll, pitch, yaw = pi / 3, pi / 4, pi / 6
        rot = rpy2matrix(roll, pitch, yaw)
        recovered_rpy = matrix2rpy(rot)
        testing.assert_almost_equal(recovered_rpy, np.array([roll, pitch, yaw]))

    def test_rotation_matrix_from_vectors(self):
        # random directions: R maps a onto b exactly
        rng = np.random.RandomState(0)
        for _ in range(50):
            a = rng.randn(3)
            b = rng.randn(3)
            rot = rotation_matrix_from_vectors(a, b)
            # valid rotation
            testing.assert_almost_equal(rot.T.dot(rot), np.eye(3))
            testing.assert_almost_equal(np.linalg.det(rot), 1.0)
            # a maps onto b (directions)
            mapped = rot.dot(normalize_vector(a))
            testing.assert_almost_equal(mapped, normalize_vector(b))

        # already-aligned -> identity
        testing.assert_almost_equal(
            rotation_matrix_from_vectors([0, 0, 2], [0, 0, 5]), np.eye(3))

        # anti-parallel -> 180 deg (maps a onto -a) and stays a valid rotation
        for a in ([0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 2, 3]):
            a = np.array(a, dtype=np.float64)
            rot = rotation_matrix_from_vectors(a, -a)
            testing.assert_almost_equal(rot.T.dot(rot), np.eye(3))
            testing.assert_almost_equal(np.linalg.det(rot), 1.0)
            testing.assert_almost_equal(
                rot.dot(normalize_vector(a)), -normalize_vector(a))

    def test_quaternion_from_vectors(self):
        rng = np.random.RandomState(1)
        for _ in range(50):
            a = rng.randn(3)
            b = rng.randn(3)
            q = quaternion_from_vectors(a, b)
            testing.assert_almost_equal(np.linalg.norm(q), 1.0)
            rot = quaternion2matrix(q)
            testing.assert_almost_equal(
                rot.dot(normalize_vector(a)), normalize_vector(b))

    def test_rpy2homogeneous(self):
        rng = np.random.RandomState(4)
        for _ in range(50):
            roll, pitch, yaw = rng.uniform(-pi, pi, 3)
            mat = rpy2homogeneous(roll, pitch, yaw)
            self.assertEqual(mat.shape, (4, 4))
            # rotation-only: zero translation, bottom row [0, 0, 0, 1]
            testing.assert_almost_equal(mat[:3, 3], np.zeros(3))
            testing.assert_almost_equal(mat[3], np.array([0, 0, 0, 1]))
            # rotation block matches rpy2matrix
            testing.assert_almost_equal(mat[:3, :3], rpy2matrix(roll, pitch, yaw))
            # consistent with xyzrpy2matrix at zero translation
            testing.assert_almost_equal(
                mat, xyzrpy2matrix([0, 0, 0], [roll, pitch, yaw]))

    def test_xyzrpy_matrix_roundtrip(self):
        rng = np.random.RandomState(2)
        for _ in range(50):
            xyz = rng.uniform(-5, 5, 3)
            rpy = np.array([rng.uniform(-pi, pi),
                            rng.uniform(-pi / 2 + 0.05, pi / 2 - 0.05),
                            rng.uniform(-pi, pi)])
            mat = xyzrpy2matrix(xyz, rpy)
            self.assertEqual(mat.shape, (4, 4))
            testing.assert_almost_equal(mat[:3, :3], rpy2matrix(*rpy))
            r_xyz, r_rpy = matrix2xyzrpy(mat)
            testing.assert_almost_equal(r_xyz, xyz)
            testing.assert_almost_equal(r_rpy, rpy)

    def test_matrix_relative(self):
        rng = np.random.RandomState(3)
        for _ in range(50):
            parent = xyzrpy2matrix(rng.uniform(-3, 3, 3), rng.uniform(-1, 1, 3))
            child = xyzrpy2matrix(rng.uniform(-3, 3, 3), rng.uniform(-1, 1, 3))
            rel = matrix_relative(parent, child)
            # matches the plain inverse, and parent @ rel reconstructs child
            testing.assert_almost_equal(rel, np.linalg.inv(parent).dot(child))
            testing.assert_almost_equal(parent.dot(rel), child)
        # relative of a frame to itself is the identity
        p = xyzrpy2matrix([1, 2, 3], [0.1, 0.2, 0.3])
        testing.assert_almost_equal(matrix_relative(p, p), np.eye(4))
