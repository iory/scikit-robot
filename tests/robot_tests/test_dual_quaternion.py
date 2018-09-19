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

    def test_screw_axis(self):
        dq = DualQuaternion()
        screw_axis, rotation, translation = dq.screw_axis()
        testing.assert_equal(
            screw_axis, np.array([0, 0, 0]))
        testing.assert_equal(
            rotation, 0)
        testing.assert_equal(
            translation, 0)
