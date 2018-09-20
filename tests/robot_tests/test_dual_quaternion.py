import math
import unittest

import numpy as np
from numpy import pi
from numpy import testing

from robot.dual_quaternion import DualQuaternion
from robot.math import normalize_vector
from robot.math import quaternion_multiply


class TestDualQuaternion(unittest.TestCase):

    def test_add(self):
        qr1 = normalize_vector(np.array([1, 2, 3, 4]))
        x, y, z = np.array([1, 2, 3])
        qd1 = 0.5 * quaternion_multiply(np.array([0, x, y, z]), qr1)
        dq1 = DualQuaternion(qr1, qd1)

        qr2 = normalize_vector(np.array([4, 3, 2, 1]))
        x, y, z = np.array([3, 2, 1])
        qd2 = 0.5 * quaternion_multiply(np.array([0, x, y, z]), qr2)
        dq2 = DualQuaternion(qr2, qd2)

        dq = (dq1 + dq2).normalize()
        testing.assert_almost_equal(
            dq.translation, [2.0, 2.6, 2.0])
        testing.assert_almost_equal(
            dq.quaternion, [0.5, 0.5, 0.5, 0.5])

    def test_mul(self):
        qr1 = normalize_vector(np.array([1, 2, 3, 4]))
        x, y, z = np.array([1, 2, 3])
        qd1 = 0.5 * quaternion_multiply(np.array([0, x, y, z]), qr1)
        dq1 = DualQuaternion(qr1, qd1)

        qr2 = normalize_vector(np.array([4, 3, 2, 1]))
        x, y, z = np.array([3, 2, 1])
        qd2 = 0.5 * quaternion_multiply(np.array([0, x, y, z]), qr2)
        dq2 = DualQuaternion(qr2, qd2)

        dq = dq1 * dq2
        testing.assert_almost_equal(
            dq.translation, [0, 4, 6])
        testing.assert_almost_equal(
            dq.quaternion, [-0.4, 0.2, 0.8, 0.4])

    def test_normalize(self):
        qr1 = normalize_vector(np.array([1, 2, 3, 4]))
        x, y, z = np.array([1, 2, 3])
        qd1 = 0.5 * quaternion_multiply(np.array([0, x, y, z]), qr1)
        dq1 = DualQuaternion(qr1, qd1)
        dq1.normalize()

        testing.assert_almost_equal(
            dq1.dq, [0.18257419, 0.36514837, 0.54772256, 0.73029674,
                     -1.82574186, 0., 0.36514837, 0.18257419])

    def test_scalar(self):
        qr = np.array([4, 1, 2, 3])
        qt = np.array([1, 1, 3, 3])
        dq = DualQuaternion(qr, qt)
        scalar = dq.scalar
        testing.assert_equal(
            scalar.dq, [4, 0, 0, 0, 1, 0, 0, 0])

    def test_screw_axis(self):
        dq = DualQuaternion()
        screw_axis, rotation, translation = dq.screw_axis()
        testing.assert_equal(
            screw_axis, np.array([0, 0, 0]))
        testing.assert_equal(
            rotation, 0)
        testing.assert_equal(
            translation, 0)

        dq = DualQuaternion([9.99961895e-01, 1.05970598e-16, 8.69655833e-03, -7.61002163e-04],
                            [-4.33680869e-18, -1.66533454e-16, -1.12628320e-04, -1.28709063e-03])
        screw_axis, rotation, translation = dq.screw_axis()
        testing.assert_almost_equal(
            screw_axis, [1.21389616e-14, 9.96193187e-01, -8.71730105e-02],
            decimal=4)
        testing.assert_almost_equal(
            rotation, 1.0003730688205559, decimal=4)
        testing.assert_almost_equal(
            translation, 9.935652945166209e-16, decimal=4)
