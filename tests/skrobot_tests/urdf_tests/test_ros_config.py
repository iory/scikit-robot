import io
import unittest
import zipfile

from skrobot.urdf.ros_config import export_all_configs
from skrobot.urdf.ros_config import generate_controllers_yaml
from skrobot.urdf.ros_config import generate_gazebo_config
from skrobot.urdf.ros_config import generate_srdf
from skrobot.urdf.ros_config import parse_urdf_content


_URDF = """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base_link"/>
  <link name="arm_link"/>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
</robot>
"""


class TestParseUrdfContent(unittest.TestCase):

    def test_parse(self):
        parsed = parse_urdf_content(_URDF)
        self.assertEqual(parsed['root_link'], 'base_link')
        self.assertEqual([j['name'] for j in parsed['joints']], ['shoulder'])
        self.assertEqual(
            sorted(link['name'] for link in parsed['links']),
            ['arm_link', 'base_link'])

    def test_invalid_xml_raises(self):
        with self.assertRaises(ValueError):
            parse_urdf_content('<not-urdf/>')
        with self.assertRaises(ValueError):
            parse_urdf_content('not xml at all <<<')


class TestGenerators(unittest.TestCase):

    def test_srdf(self):
        srdf = generate_srdf(
            'two_link',
            [{'name': 'arm', 'joints': ['shoulder']}],
            [('base_link', 'arm_link')])
        self.assertIn('<robot name="two_link">', srdf)
        self.assertIn('arm', srdf)
        self.assertIn('disable_collisions', srdf)

    def test_controllers_yaml(self):
        yaml_text = generate_controllers_yaml(
            [{'name': 'arm_controller',
              'type': 'joint_trajectory_controller',
              'joints': ['shoulder']}])
        self.assertIn('shoulder', yaml_text)
        # ros2_control resolves controller types from the controller
        # manager's parameter namespace: the type must be nested there.
        self.assertIn(
            '    arm_controller:\n'
            '      type: joint_trajectory_controller/JointTrajectoryController',
            yaml_text)
        # the controller's own node section carries its joints
        self.assertIn(
            'arm_controller:\n'
            '  ros__parameters:\n'
            '    joints:\n'
            '      - shoulder',
            yaml_text)

    def test_ros2_control_xacro_package_name(self):
        from skrobot.urdf.ros_config.gazebo_generator import generate_ros2_control_xacro
        joints = [{'name': 'shoulder', 'type': 'revolute'}]
        self.assertIn('$(find robot_config)/config/controllers.yaml',
                      generate_ros2_control_xacro(joints))
        self.assertIn('$(find my_bot)/config/controllers.yaml',
                      generate_ros2_control_xacro(joints,
                                                  package_name='my_bot'))

    def test_gazebo_config(self):
        text = generate_gazebo_config({'gravity': [0, 0, -9.81]}, [])
        self.assertTrue(text.strip())


class TestExportAllConfigs(unittest.TestCase):

    def test_zip_bundle(self):
        parsed = parse_urdf_content(_URDF)
        blob = export_all_configs(
            urdf_content=_URDF,
            joints=parsed['joints'],
            planning_groups=[{'name': 'arm', 'joints': ['shoulder']}],
            controllers=[{'name': 'arm_controller', 'type': 'position',
                          'joints': ['shoulder']}],
            disabled_collision_pairs=[('base_link', 'arm_link')],
            gazebo_physics={'gravity': [0, 0, -9.81]},
            gazebo_plugins=[],
            robot_name='two_link',
            extra_files={'two_link/config/servo_mapping.yaml': 'servos: []'})
        self.assertIsInstance(blob, bytes)
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            names = archive.namelist()
            self.assertTrue(names)
            # the bundle must at least carry the URDF itself
            self.assertTrue(any(name.endswith('.urdf') for name in names),
                            names)
            # caller-provided extra files are written verbatim
            self.assertIn('two_link/config/servo_mapping.yaml', names)
            # the ros2_control plugin must point at THIS package, not a
            # hard-coded one
            xacro = archive.read(
                'two_link/urdf/ros2_control.xacro').decode('utf-8')
            self.assertIn('$(find two_link)/config/controllers.yaml', xacro)


if __name__ == '__main__':
    unittest.main()
