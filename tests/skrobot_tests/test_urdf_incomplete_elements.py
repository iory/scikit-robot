"""Elements a URDF leaves half-written, which other parsers tolerate.

Each of these is malformed by the spec and ordinary in practice, and
each one used to raise from deep inside the parser and take the whole
robot with it. None of them describes kinematics: a texture, a material
name, an inertia, a gear ratio nothing reads. Losing the model over any
of them costs far more than the element was worth.
"""

import unittest

from skrobot.models.urdf import RobotModelFromURDF


def urdf(link_extra='', extra=''):
    """Wrap the fragment under test in a robot that would load without it."""
    return """<?xml version="1.0"?>
<robot name="incomplete">
  <link name="base">
    {}
  </link>
  <link name="tip"/>
  <joint name="elbow" type="revolute">
    <parent link="base"/>
    <child link="tip"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  {}
</robot>""".format(link_extra, extra)


class TestMaterialWithoutName(unittest.TestCase):
    """``<material>`` carrying its colour inline needs no name.

    The spec says the attribute is required -- it is how one material
    refers to another -- but a material that states its own colour
    refers to nothing, and models leave the name out.
    """

    def test_model_loads(self):
        robot = RobotModelFromURDF(urdf(link_extra="""
    <visual>
      <geometry><box size="1 1 1"/></geometry>
      <material><color rgba="1 0 0 1"/></material>
    </visual>"""))
        self.assertEqual(['base', 'tip'],
                         sorted(link.name for link in robot.link_list))

    def test_colour_survives(self):
        # The point of an unnamed material is the colour it carries. The
        # loader already names such a material for itself once it is
        # built -- it never got that far, because the constructor
        # refused to build one at all.
        robot = RobotModelFromURDF(urdf(link_extra="""
    <visual>
      <geometry><box size="1 1 1"/></geometry>
      <material><color rgba="1 0 0 1"/></material>
    </visual>"""))
        material = robot.urdf_robot_model.links[0].visuals[0].material
        self.assertAlmostEqual(1.0, material.color[0])
        self.assertAlmostEqual(0.0, material.color[1])
        self.assertTrue(material.name.startswith('__unnamed_material'))


class TestTextureWithoutFilename(unittest.TestCase):
    """``<texture>`` with no file named is decoration, not kinematics."""

    def test_model_loads(self):
        robot = RobotModelFromURDF(urdf(link_extra="""
    <visual>
      <geometry><box size="1 1 1"/></geometry>
      <material name="painted"><texture/></material>
    </visual>"""))
        self.assertEqual(2, len(robot.link_list))


class TestIncompleteInertial(unittest.TestCase):
    """``<inertial>`` that stops after the mass, or omits terms.

    Hand-written descriptions do this constantly. The link still has a
    shape and a place in the tree, which is what most callers want from
    it.
    """

    def test_inertial_without_inertia(self):
        robot = RobotModelFromURDF(urdf(link_extra="""
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="2.5"/>
    </inertial>"""))
        self.assertEqual(2, len(robot.link_list))
        inertial = robot.urdf_robot_model.links[0].inertial
        self.assertAlmostEqual(2.5, inertial.mass)

    def test_inertial_without_mass(self):
        robot = RobotModelFromURDF(urdf(link_extra="""
    <inertial>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>"""))
        self.assertAlmostEqual(0.0, robot.urdf_robot_model.links[0].inertial.mass)

    def test_inertia_missing_terms(self):
        # Only the diagonal given; the products of inertia are zero.
        robot = RobotModelFromURDF(urdf(link_extra="""
    <inertial>
      <mass value="1"/>
      <inertia ixx="2" iyy="3" izz="4"/>
    </inertial>"""))
        inertia = robot.urdf_robot_model.links[0].inertial.inertia
        self.assertAlmostEqual(2.0, inertia[0][0])
        self.assertAlmostEqual(0.0, inertia[0][1])


class TestUnusableMechanicalReduction(unittest.TestCase):
    """``<mechanicalReduction>`` holding a word instead of a ratio.

    Package templates ship the placeholder and nobody replaces it. The
    value is ros_control metadata that nothing here reads.
    """

    def test_model_loads(self):
        robot = RobotModelFromURDF(urdf(extra="""
  <transmission name="elbow_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="elbow"><hardwareInterface>Effort</hardwareInterface></joint>
    <actuator name="elbow_motor">
      <mechanicalReduction>reduction</mechanicalReduction>
    </actuator>
  </transmission>"""))
        self.assertEqual(['elbow'], [j.name for j in robot.joint_list])

    def test_unusable_value_is_dropped(self):
        robot = RobotModelFromURDF(urdf(extra="""
  <transmission name="elbow_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="elbow"><hardwareInterface>Effort</hardwareInterface></joint>
    <actuator name="elbow_motor">
      <mechanicalReduction>reduction</mechanicalReduction>
    </actuator>
  </transmission>"""))
        actuator = robot.urdf_robot_model.transmissions[0].actuators[0]
        self.assertIsNone(actuator.mechanicalReduction)


if __name__ == '__main__':
    unittest.main()
