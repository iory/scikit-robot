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

    def test_joint_keys_are_snake_case(self):
        joint = parse_urdf_content(_URDF)['joints'][0]
        self.assertEqual(
            sorted(joint),
            ['axis', 'child_link', 'effort_limit', 'is_mimic', 'lower_limit',
             'mimic_joint', 'name', 'parent_link', 'type', 'upper_limit',
             'velocity_limit'])
        self.assertEqual(joint['parent_link'], 'base_link')
        self.assertEqual(joint['child_link'], 'arm_link')
        self.assertEqual(joint['lower_limit'], -1.5)
        self.assertEqual(joint['upper_limit'], 1.5)
        self.assertEqual(joint['velocity_limit'], 1.0)
        self.assertEqual(joint['effort_limit'], 10.0)
        self.assertFalse(joint['is_mimic'])
        self.assertIsNone(joint['mimic_joint'])

    def test_link_keys_are_snake_case(self):
        link = parse_urdf_content(_URDF)['links'][0]
        self.assertEqual(
            sorted(link),
            ['has_collision', 'has_inertial', 'has_visual', 'name'])

    def test_mimic_joint_is_reported(self):
        urdf = _URDF.replace(
            '</robot>',
            '  <joint name="follower" type="revolute">\n'
            '    <parent link="arm_link"/>\n'
            '    <child link="tip_link"/>\n'
            '    <mimic joint="shoulder" multiplier="1" offset="0"/>\n'
            '  </joint>\n'
            '  <link name="tip_link"/>\n'
            '</robot>')
        joints = {j['name']: j for j in parse_urdf_content(urdf)['joints']}
        self.assertTrue(joints['follower']['is_mimic'])
        self.assertEqual(joints['follower']['mimic_joint'], 'shoulder')


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

    def test_ros2_control_xacro_targets_gazebo_sim(self):
        from skrobot.urdf.ros_config.gazebo_generator import generate_ros2_control_xacro
        xacro = generate_ros2_control_xacro(
            [{'name': 'shoulder', 'type': 'revolute'}])
        self.assertIn('gz_ros2_control/GazeboSimSystem', xacro)
        self.assertIn('libgz_ros2_control-system.so', xacro)
        # Gazebo Classic and its gazebo_ros2_control plugin reached end
        # of life in January 2025; only gz_ros2_control is emitted.
        self.assertNotIn('gazebo_ros2_control/GazeboSystem', xacro)
        self.assertNotIn('libgazebo_ros2_control.so', xacro)

    def test_srdf_end_effector_link(self):
        srdf = generate_srdf(
            'two_link',
            [{'name': 'arm', 'joints': ['shoulder'],
              'end_effector_link': 'arm_link'}],
            [])
        self.assertIn(
            '<end_effector name="arm_eef" parent_link="arm_link" '
            'group="arm" />',
            srdf)

    def test_ros2_control_xacro_skips_mimic_joints(self):
        from skrobot.urdf.ros_config.gazebo_generator import generate_ros2_control_xacro
        xacro = generate_ros2_control_xacro([
            {'name': 'shoulder', 'type': 'revolute'},
            {'name': 'follower', 'type': 'revolute', 'is_mimic': True},
            {'name': 'weld', 'type': 'fixed'},
        ])
        self.assertIn('<joint name="shoulder">', xacro)
        # a mimic joint is constrained by the URDF <mimic> tag; giving it
        # a command interface would fight that constraint
        self.assertNotIn('follower', xacro)
        self.assertNotIn('weld', xacro)

    def test_gazebo_config(self):
        text = generate_gazebo_config({'gravity': [0, 0, -9.81]}, [])
        self.assertTrue(text.strip())

    def test_gazebo_physics_settings(self):
        text = generate_gazebo_config(
            {'solver': 'bullet', 'step_size': 0.002,
             'real_time_factor': 0.5},
            [])
        self.assertIn('<physics name="default_physics" type="bullet">', text)
        self.assertIn('<max_step_size>0.002</max_step_size>', text)
        self.assertIn('<real_time_factor>0.5</real_time_factor>', text)
        self.assertIn('<real_time_update_rate>500</real_time_update_rate>',
                      text)


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

    def test_export_options_select_sections(self):
        parsed = parse_urdf_content(_URDF)
        blob = export_all_configs(
            urdf_content=_URDF,
            joints=parsed['joints'],
            planning_groups=[{'name': 'arm', 'joints': ['shoulder']}],
            controllers=[],
            disabled_collision_pairs=[],
            gazebo_physics={},
            gazebo_plugins=[],
            robot_name='two_link',
            export_options={'include_urdf': True,
                            'include_moveit': False,
                            'include_gazebo': False})
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            names = archive.namelist()
            self.assertIn('two_link/urdf/two_link.urdf', names)
            self.assertNotIn('two_link/config/two_link.srdf', names)
            self.assertNotIn('two_link/urdf/ros2_control.xacro', names)

    def test_zip_bundle_rewrites_mesh_package(self):
        # mesh references to a foreign package must be rewritten to the
        # bundle's own package name so the archive is self-contained
        urdf = (
            '<?xml version="1.0"?>\n'
            '<robot name="two_link">\n'
            '  <link name="base_link">\n'
            '    <visual><geometry>'
            '<mesh filename="package://foreign_pkg/meshes/base.stl"/>'
            '</geometry></visual>\n'
            '  </link>\n'
            '</robot>\n')
        blob = export_all_configs(
            urdf_content=urdf,
            joints=[],
            planning_groups=[],
            controllers=[],
            disabled_collision_pairs=[],
            gazebo_physics={'gravity': [0, 0, -9.81]},
            gazebo_plugins=[],
            robot_name='two_link')
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            urdf_out = archive.read(
                'two_link/urdf/two_link.urdf').decode('utf-8')
            self.assertIn('package://two_link/meshes/base.stl', urdf_out)
            self.assertNotIn('foreign_pkg', urdf_out)


if __name__ == '__main__':
    unittest.main()
