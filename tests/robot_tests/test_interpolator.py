import unittest

from numpy import pi
from numpy import testing
import numpy as np

from robot.interpolator import LinearInterpolator
from robot.interpolator import MinjerkInterpolator
from robot.math import midpoint


class TestInterpolator(unittest.TestCase):

    def test_linear_interpolator(self):
        ip = LinearInterpolator()
        p0 = np.array([1, 2, 3])
        t0 = 0.10
        p1 = np.array([3, 4, 5])
        t1 = 0.18
        ip.reset(position_list=[p0, p1, p0],
                 time_list=[t0, t1])
        ip.start_interpolation()

        i = 0
        while t0 > i:
            ip.pass_time(0.02)
            testing.assert_almost_equal(ip.position, midpoint(i / t0, p0, p1))
            i += 0.02

        i = t0
        while t1 > i:
            ip.pass_time(0.02)
            testing.assert_almost_equal(
                ip.position,
                midpoint((i - t0) / (t1 - t0), p1, p0))
            i += 0.02

        assert(ip.is_interpolating is False)

    def test_minjerk_interpolator(self):
        ip = MinjerkInterpolator()
        p0 = np.array([1, 2, 3])
        t0 = 0.10
        p1 = np.array([3, 4, 5])
        t1 = 0.18
        ip.reset(position_list=[p0, p1, p0],
                 time_list=[t0, t1])
        ip.start_interpolation()

        i = 0
        while t0 >= i:
            ip.pass_time(0.02)
            i += 0.02
        testing.assert_almost_equal(ip.position, p1)

        i = t0
        while t1 >= i:
            ip.pass_time(0.02)
            i += 0.02
        testing.assert_almost_equal(ip.position, p0)

        assert(ip.is_interpolating is False)
