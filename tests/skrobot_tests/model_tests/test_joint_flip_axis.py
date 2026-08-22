import unittest

import numpy as np

from skrobot.model import FixedJoint
from skrobot.model import FloatingJoint
from skrobot.model import LinearJoint
from skrobot.model import Link
from skrobot.model import OmniWheelJoint
from skrobot.model import PlanarJoint
from skrobot.model import RotationalJoint
from skrobot.model.joint_limit_table import JointLimitTable


def _arm(cls=RotationalJoint, axis='z', min_angle=-np.pi, max_angle=np.pi):
    """A parent link with one joint driving a child link 10cm away."""
    base = Link(name='base_link')
    arm = Link(name='arm_link', pos=[0.1, 0.0, 0.0])
    joint = cls(name='arm_joint', parent_link=base, child_link=arm,
                axis=axis, min_angle=min_angle, max_angle=max_angle)
    base.add_child_link(arm)
    arm.add_parent_link(base)
    arm.add_joint(joint)
    base.assoc(arm, relative_coords='local')
    return base, arm, joint


class TestJointFlipAxis(unittest.TestCase):

    def test_it_negates_the_axis_and_swaps_the_limits(self):
        _base, _child, joint = _arm(min_angle=-0.5, max_angle=2.0)

        self.assertTrue(joint.flip_axis())

        np.testing.assert_allclose(joint.axis, [0, 0, -1], atol=1e-12)
        self.assertEqual(joint.min_angle, -2.0)
        self.assertEqual(joint.max_angle, 0.5)

    def test_a_posed_joint_does_not_move(self):
        """The point of the method: the sign changes, the robot does not."""
        _base, arm, joint = _arm()
        joint.joint_angle(0.7)
        before_pos = arm.worldpos().copy()
        before_rot = arm.worldrot().copy()

        joint.flip_axis()

        np.testing.assert_allclose(arm.worldpos(), before_pos, atol=1e-9)
        np.testing.assert_allclose(arm.worldrot(), before_rot, atol=1e-9)
        self.assertEqual(joint.joint_angle(), -0.7)

    def test_a_range_that_excludes_zero_survives_the_flip(self):
        """The value passes through zero on the way and the limits forbid it.

        ``joint_angle`` clamps rather than refuses, so a flip that did not
        open the limits for the trip would leave the arm at 0.5 rad and
        report it as if the user had asked for that.
        """
        _base, arm, joint = _arm(min_angle=0.5, max_angle=1.0)
        joint.joint_angle(0.8)
        before_pos = arm.worldpos().copy()

        joint.flip_axis()

        self.assertEqual(joint.joint_angle(), -0.8)
        self.assertEqual((joint.min_angle, joint.max_angle), (-1.0, -0.5))
        np.testing.assert_allclose(arm.worldpos(), before_pos, atol=1e-9)

    def test_flipping_twice_gives_the_joint_back(self):
        _base, arm, joint = _arm(axis='y', min_angle=-0.5, max_angle=1.2)
        joint.joint_angle(0.3)
        before_pos = arm.worldpos().copy()

        joint.flip_axis()
        joint.flip_axis()

        np.testing.assert_allclose(joint.axis, [0, 1, 0], atol=1e-12)
        self.assertEqual((joint.min_angle, joint.max_angle), (-0.5, 1.2))
        self.assertEqual(joint.joint_angle(), 0.3)
        np.testing.assert_allclose(arm.worldpos(), before_pos, atol=1e-9)

    def test_a_continuous_joint_keeps_its_unbounded_range(self):
        _base, _child, joint = _arm(min_angle=-np.inf, max_angle=np.inf)

        self.assertTrue(joint.flip_axis())

        self.assertEqual(joint.min_angle, -np.inf)
        self.assertEqual(joint.max_angle, np.inf)

    def test_a_linear_joint_flips_the_same_way(self):
        _base, arm, joint = _arm(cls=LinearJoint, min_angle=-0.2, max_angle=0.5)
        joint.joint_angle(0.3)
        before_pos = arm.worldpos().copy()

        self.assertTrue(joint.flip_axis())

        np.testing.assert_allclose(joint.axis, [0, 0, -1], atol=1e-12)
        self.assertEqual((joint.min_angle, joint.max_angle), (-0.5, 0.2))
        self.assertEqual(joint.joint_angle(), -0.3)
        np.testing.assert_allclose(arm.worldpos(), before_pos, atol=1e-9)

    def test_a_joint_with_no_single_direction_is_left_alone(self):
        """A fixed joint has no direction; a multi-DOF one has several."""
        base, arm = Link(name='p'), Link(name='c')
        self.assertFalse(
            FixedJoint(parent_link=base, child_link=arm).flip_axis())
        for cls in (PlanarJoint, FloatingJoint, OmniWheelJoint):
            joint = cls(parent_link=Link(name='p'), child_link=Link(name='c'))
            self.assertFalse(joint.flip_axis(),
                             '{} has no single axis'.format(cls.__name__))

    def test_a_mimic_follower_is_not_dragged_along(self):
        """Flipping a leader must not move the joints hooked to it.

        The follower is already in the pose the mechanism holds; the
        coupling is what needs re-signing, and that is the caller's job.
        """
        _base, _child, leader = _arm()
        _fbase, follower_link, follower = _arm()
        leader.joint_angle(0.5)
        follower.joint_angle(1.0)
        leader.register_mimic_joint(follower, multiplier=2.0, offset=0.0)
        follower_pos = follower_link.worldpos().copy()

        leader.flip_axis()

        self.assertEqual(follower.joint_angle(), 1.0)
        np.testing.assert_allclose(
            follower_link.worldpos(), follower_pos, atol=1e-9)

    def test_a_joint_limit_table_is_refused_rather_than_contradicted(self):
        _base, _child, joint = _arm()
        _tbase, _tchild, target = _arm()
        joint.joint_min_max_table = JointLimitTable(
            target, -np.pi / 2.0, np.pi / 2.0,
            np.array([-1.0, -0.5]), np.array([1.0, 0.5]))
        joint.joint_min_max_target = target

        with self.assertRaises(ValueError):
            joint.flip_axis()

        # Nothing was changed on the way to refusing.
        np.testing.assert_allclose(joint.axis, [0, 0, 1], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
