import unittest

from skrobot.model import FixedJoint
from skrobot.model import FloatingJoint
from skrobot.model import LinearJoint
from skrobot.model import Link
from skrobot.model import OmniWheelJoint
from skrobot.model import PlanarJoint
from skrobot.model import RotationalJoint
from skrobot.urdf.structure import MOVABLE_JOINT_TYPES


def _joint(cls):
    return cls(parent_link=Link(name='p'), child_link=Link(name='c'))


class TestJointIsMovable(unittest.TestCase):

    def test_only_a_fixed_joint_is_immovable(self):
        for cls in (RotationalJoint, LinearJoint, PlanarJoint,
                    FloatingJoint, OmniWheelJoint):
            self.assertTrue(_joint(cls).is_movable,
                            '{} moves'.format(cls.__name__))
        self.assertFalse(_joint(FixedJoint).is_movable)

    def test_it_follows_the_degrees_of_freedom(self):
        for cls in (RotationalJoint, LinearJoint, PlanarJoint,
                    FloatingJoint, OmniWheelJoint, FixedJoint):
            joint = _joint(cls)
            self.assertEqual(joint.is_movable, joint.joint_dof > 0)

    def test_it_classifies_a_joint_that_has_no_type(self):
        """OmniWheelJoint carries three degrees of freedom and no ``type``.

        Anything deciding this by comparing type names raises instead of
        answering, which is the reason not to decide it that way.
        """
        joint = _joint(OmniWheelJoint)
        with self.assertRaises(NotImplementedError):
            joint.type
        self.assertTrue(joint.is_movable)

    def test_it_agrees_with_the_urdf_type_list(self):
        """The same question is asked of URDF type strings elsewhere.

        ``skrobot.urdf.structure`` has only the type name to go on, so it
        keeps a list. The two must not drift apart for the joints that
        both can see.
        """
        for cls in (RotationalJoint, LinearJoint, PlanarJoint, FloatingJoint):
            joint = _joint(cls)
            self.assertIn(joint.type, MOVABLE_JOINT_TYPES)
        self.assertNotIn(_joint(FixedJoint).type, MOVABLE_JOINT_TYPES)


if __name__ == '__main__':
    unittest.main()
