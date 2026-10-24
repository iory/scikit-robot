import unittest
from unittest import mock

from skrobot.model import LinearJoint
from skrobot.model import Link
from skrobot.model import RotationalJoint


def _joint(cls):
    return cls(parent_link=Link(name='p'), child_link=Link(name='c'),
               min_angle=-1.57, max_angle=1.57)


class TestJointLimitRounding(unittest.TestCase):
    """A joint angle a rounding error past its limit is clamped quietly.

    1.5700000524 is what 1.57 becomes through float32; warning about it
    fills a log with noise from any solver working in single precision.
    """

    def test_a_rounding_error_is_clamped_without_a_warning(self):
        for cls in (RotationalJoint, LinearJoint):
            for v, limit in ((1.57 + 5e-8, 1.57), (-1.57 - 5e-8, -1.57)):
                joint = _joint(cls)
                with mock.patch('skrobot.model.joint.logger') as logger:
                    joint.joint_angle(v)
                logger.warning.assert_not_called()
                self.assertEqual(joint.joint_angle(), limit)

    def test_a_real_violation_is_still_clamped_and_warned(self):
        for cls in (RotationalJoint, LinearJoint):
            for v, limit in ((1.57 + 1e-3, 1.57), (-1.57 - 1e-3, -1.57)):
                joint = _joint(cls)
                with mock.patch('skrobot.model.joint.logger') as logger:
                    joint.joint_angle(v)
                logger.warning.assert_called_once()
                self.assertIn('violate', logger.warning.call_args[0][0])
                self.assertEqual(joint.joint_angle(), limit)
