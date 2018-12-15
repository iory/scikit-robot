import unittest

import numpy as np
from numpy import testing

from robot.quaternion import Quaternion


class TestQuaternion(unittest.TestCase):

    def test_init(self):
        q = [1, 0, 0, 0]
        Quaternion(q)
        Quaternion(q=q)
        Quaternion(q[0], q[1], q[2], q[3])

    def test_axis(self):
        q = Quaternion(w=0.7071, x=0.7071)
        testing.assert_almost_equal(
            q.axis, [1.0, 0.0, 0.0])

        q = Quaternion(w=0.7071, y=0.7071)
        testing.assert_almost_equal(
            q.axis, [0.0, 1.0, 0.0])

        q = Quaternion(w=0.7071, z=0.7071)
        testing.assert_almost_equal(
            q.axis, [0.0, 0.0, 1.0])

        q = Quaternion([1, 1, 1, 1])
        testing.assert_almost_equal(
            q.axis, [0.5773503, 0.5773503, 0.5773503])

    def test_angle(self):
        q = Quaternion(w=0.7071, x=0.7071)
        testing.assert_almost_equal(
            q.angle, np.pi / 2.0)

        q = Quaternion(w=0.7071, y=0.7071)
        testing.assert_almost_equal(
            q.angle, np.pi / 2.0)

        q = Quaternion(w=0.7071, z=0.7071)
        testing.assert_almost_equal(
            q.angle, np.pi / 2.0)

        q = Quaternion([1, 1, 1, 1])
        testing.assert_almost_equal(
            q.angle, 2.0943951)

    def test_add(self):
        q1 = Quaternion()
        q2 = Quaternion()

        q = q1 + q2
        testing.assert_almost_equal(
            q.q, [2.0, 0.0, 0.0, 0.0])

    def test_mul(self):
        q1 = Quaternion()
        q2 = Quaternion()

        q = q1 * q2
        testing.assert_almost_equal(
            q.q, [1.0, 0.0, 0.0, 0.0])

        q = 3.0 * q1
        testing.assert_almost_equal(
            q.q, [3.0, 0.0, 0.0, 0.0])

    def test_div(self):
        q1 = Quaternion()
        q2 = Quaternion()

        q = q1 / q2
        testing.assert_almost_equal(
            q.q, [1.0, 0.0, 0.0, 0.0])

        q = q1 / 2.0
        testing.assert_almost_equal(
            q.q, [0.5, 0.0, 0.0, 0.0])

    def test_norm(self):
        q = Quaternion()

        testing.assert_almost_equal(
            q.norm, 1.0)

    def test_conjugate(self):
        q = Quaternion()
        testing.assert_almost_equal(
            q.conjugate().q,
            [1.0, 0.0, 0.0, 0.0])

        q = Quaternion(q=[1, -1, -2, -3])
        testing.assert_almost_equal(
            q.conjugate().q,
            [1, 1, 2, 3])

    def test_inverse(self):
        q = Quaternion()
        testing.assert_almost_equal(
            q.inverse().q,
            [1.0, 0.0, 0.0, 0.0])
