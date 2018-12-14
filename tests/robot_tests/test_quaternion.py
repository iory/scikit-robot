import unittest

from numpy import testing

from robot.quaternion import Quaternion


class TestQuaternion(unittest.TestCase):

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
            q.norm(), 1.0)

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
