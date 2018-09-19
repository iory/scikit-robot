import math
import unittest

import numpy as np
from numpy import pi
from numpy import testing

from robot.dual_quaternion import DualQuaternion


class TestDualQuaternion(unittest.TestCase):

    def test_screw_axis(self):
        dq = DualQuaternion()
        screw_axis, rotation, translation = dq.screw_axis()
        testing.assert_equal(
            screw_axis, np.array([0, 0, 0]))
        testing.assert_equal(
            rotation, 0)
        testing.assert_equal(
            translation, 0)
