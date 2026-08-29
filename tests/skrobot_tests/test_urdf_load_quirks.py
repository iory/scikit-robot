"""URDF constructs that other parsers accept but that aborted the load here.

Each case below comes from a published robot description. None of them is
exotic, and every one of them raised before reaching a usable model.
"""

import unittest

from skrobot.model import Link
from skrobot.models.urdf import RobotModelFromURDF


def urdf(joints, extra=''):
    """Wrap link/joint XML in a minimal robot."""
    return """<?xml version="1.0"?>
<robot name="quirky">
  <link name="base"/>
  <link name="body"/>
  <link name="tip"/>
  {}
  {}
</robot>""".format(joints, extra)


REVOLUTE = """
  <joint name="elbow" type="revolute">
    <parent link="body"/>
    <child link="tip"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>"""


class TestUnsupportedJointType(unittest.TestCase):
    """A floating base must not take the whole model down with it.

    ``floating`` and ``planar`` have no counterpart among the joint
    classes, so the load left the joint object unbuilt and then raised
    ``UnboundLocalError`` on it. Quadrupeds and drones ship a floating
    base routinely.
    """

    def load(self, joint_type):
        joints = """
  <joint name="root_joint" type="{}">
    <parent link="base"/>
    <child link="body"/>
  </joint>""".format(joint_type) + REVOLUTE
        return RobotModelFromURDF(urdf=urdf(joints))

    def test_floating_joint_loads(self):
        robot = self.load('floating')
        self.assertEqual(['base', 'body', 'tip'],
                         sorted(link.name for link in robot.link_list))

    def test_unsupported_joint_is_not_actuated(self):
        # It carries no angle we can set, so it has no place among the
        # joints the model drives.
        robot = self.load('floating')
        self.assertEqual(['elbow'], [j.name for j in robot.joint_list])

    def test_unsupported_joint_keeps_the_tree_whole(self):
        # Dropping the joint instead would orphan every link below it.
        robot = self.load('planar')
        self.assertIs(robot.base, robot.body.parent_link)
        self.assertIs(robot.body, robot.tip.parent_link)

    def test_unsupported_joint_is_reported(self):
        with self.assertLogs('skrobot.model.robot_model', level='WARNING') \
                as captured:
            self.load('floating')
        self.assertIn('floating', '\n'.join(captured.output))


class TestNameSharedByLinkAndJoint(unittest.TestCase):
    """URDF names links and joints in separate namespaces.

    A joint may therefore carry the base link's name. Both were written
    into the model's attributes under that one name, the joint last, so
    the base link lookup returned a joint and the load raised.
    """

    def load(self):
        joints = """
  <joint name="base" type="fixed">
    <parent link="base"/>
    <child link="body"/>
  </joint>""" + REVOLUTE
        return RobotModelFromURDF(urdf=urdf(joints))

    def test_root_link_is_a_link(self):
        robot = self.load()
        self.assertIsInstance(robot.root_link, Link)
        self.assertEqual('base', robot.root_link.name)

    def test_forward_kinematics_runs(self):
        robot = self.load()
        robot.elbow.joint_angle(0.5)
        self.assertTrue(robot.tip.worldcoords().translation.shape == (3,))


class TestActuatorWithoutName(unittest.TestCase):
    """``<actuator>`` is ros_control metadata, not kinematics.

    The spec makes its name mandatory and models omit it anyway. Nothing
    here reads the name, so its absence must not abort the load.
    """

    def load(self):
        transmission = """
  <transmission name="elbow_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="elbow">
      <hardwareInterface>EffortJointInterface</hardwareInterface>
    </joint>
    <actuator>
      <mechanicalReduction>25</mechanicalReduction>
    </actuator>
  </transmission>"""
        joints = """
  <joint name="waist" type="fixed">
    <parent link="base"/>
    <child link="body"/>
  </joint>""" + REVOLUTE
        return RobotModelFromURDF(urdf=urdf(joints, transmission))

    def test_model_loads(self):
        robot = self.load()
        self.assertEqual(['elbow'], [j.name for j in robot.joint_list])

    def test_missing_name_stays_missing(self):
        robot = self.load()
        actuator = robot.urdf_robot_model.transmissions[0].actuators[0]
        self.assertIsNone(actuator.name)


if __name__ == '__main__':
    unittest.main()
