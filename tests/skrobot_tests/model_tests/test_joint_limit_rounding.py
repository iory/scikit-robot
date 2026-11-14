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
                self.assertIn('exceeds', logger.warning.call_args[0][0])
                self.assertEqual(joint.joint_angle(), limit)


def test_limit_warning_names_joint_and_shows_both_units(caplog):
    cases = (
        (RotationalJoint, 0.5235987755982988, 0.52,
         'rarm_joint1: 0.5236 rad (30.0 deg) exceeds max 0.5200 rad '
         '(29.8 deg); clipped to max. range [-0.5200, 0.5200] rad '
         '([-29.8, 29.8] deg)'),
        (LinearJoint, -0.05, 0.12,
         'rarm_joint1: -0.0500 m (-50.0 mm) exceeds min 0.0000 m (0.0 mm); '
         'clipped to min. range [0.0000, 0.1200] m ([0.0, 120.0] mm)'),
    )
    for cls, v, hi, expected in cases:
        lo = -hi if cls is RotationalJoint else 0.0
        joint = cls(name='rarm_joint1', parent_link=Link(name='p'),
                    child_link=Link(name='c'), min_angle=lo, max_angle=hi)
        caplog.clear()
        with caplog.at_level('WARNING', logger='skrobot.model.joint'):
            joint.joint_angle(v)
        assert [r.getMessage() for r in caplog.records] == [expected]
