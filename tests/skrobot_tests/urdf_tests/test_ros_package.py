import unittest
import xml.etree.ElementTree as ET

from skrobot.urdf.ros_package import extract_all_resource_references
from skrobot.urdf.ros_package import extract_mesh_references
from skrobot.urdf.ros_package import extract_registered_mesh_references
from skrobot.urdf.ros_package import extract_xacro_includes
from skrobot.urdf.ros_package import generate_cmake_lists
from skrobot.urdf.ros_package import generate_package_xml
from skrobot.urdf.ros_package import generate_ros1_display_cmake_lists
from skrobot.urdf.ros_package import generate_ros1_display_launch
from skrobot.urdf.ros_package import generate_ros1_display_package_xml
from skrobot.urdf.ros_package import generate_ros1_rviz_config
from skrobot.urdf.ros_package import replace_package_references


class TestPackageSkeleton(unittest.TestCase):

    def test_package_xml_is_valid_and_substituted(self):
        xml = generate_package_xml('my_bot', version='2.1.0',
                                   maintainer='Alice',
                                   maintainer_email='a@example.com',
                                   license_name='BSD')
        root = ET.fromstring(xml)
        self.assertEqual(root.find('name').text, 'my_bot')
        self.assertEqual(root.find('version').text, '2.1.0')
        self.assertEqual(root.find('license').text, 'BSD')
        self.assertEqual(root.find('maintainer').get('email'),
                         'a@example.com')

    def test_ros1_display_package_xml_has_display_deps(self):
        xml = generate_ros1_display_package_xml('my_bot')
        root = ET.fromstring(xml)
        exec_deps = {el.text for el in root.findall('exec_depend')}
        self.assertIn('robot_state_publisher', exec_deps)
        self.assertIn('joint_state_publisher_gui', exec_deps)
        self.assertIn('rviz', exec_deps)

    def test_cmake_lists_substituted(self):
        for gen in (generate_cmake_lists, generate_ros1_display_cmake_lists):
            text = gen('my_bot')
            self.assertIn('project(my_bot)', text)
            self.assertIn('${CATKIN_PACKAGE_SHARE_DESTINATION}', text)
        self.assertIn('DIRECTORY launch/',
                      generate_ros1_display_cmake_lists('my_bot'))

    def test_cmake_lists_without_xacro(self):
        text = generate_cmake_lists('my_bot', include_xacro=False)
        self.assertNotIn('xacro', text)
        self.assertIn('DIRECTORY meshes/', text)

    def test_display_launch_is_valid_xml(self):
        launch = generate_ros1_display_launch('my_bot')
        root = ET.fromstring(launch)
        self.assertEqual(root.tag, 'launch')
        self.assertIn('$(find my_bot)/urdf/my_bot.urdf', launch)

    def test_rviz_config_nonempty(self):
        self.assertIn('rviz', generate_ros1_rviz_config().lower())


class TestResourceReferences(unittest.TestCase):

    _URDF = (
        '<robot>'
        '<mesh filename="package://pkg/meshes/visual/a.stl"/>'
        '<mesh filename="file:///tmp/x/meshes/b.dae"/>'
        '<mesh filename="package://pkg/materials/textures/t.png"/>'
        '<mesh filename="package://pkg/registered/h/mod/meshes/c.stl"/>'
        '</robot>')

    def test_mesh_references(self):
        refs = extract_mesh_references(self._URDF)
        self.assertIn('visual/a.stl', refs)
        self.assertIn('b.dae', refs)

    def test_all_resource_references_excludes_registered(self):
        refs = extract_all_resource_references(self._URDF)
        self.assertIn('materials/textures/t.png', refs)
        self.assertIn('meshes/visual/a.stl', refs)
        self.assertFalse(any(r.startswith('registered/') for r in refs))

    def test_registered_mesh_references(self):
        refs = extract_registered_mesh_references(self._URDF)
        self.assertEqual(refs, {'registered/h/mod/meshes/c.stl'})

    def test_xacro_includes(self):
        xacro = ('<xacro:include filename="$(find pkg)/xacro/arm.xacro"/>'
                 '<xacro:include filename="/abs/path/xacro/leg.xacro"/>')
        self.assertEqual(extract_xacro_includes(xacro),
                         {'arm.xacro', 'leg.xacro'})

    def test_replace_package_references(self):
        out = replace_package_references(
            'package://old_pkg/meshes/a.stl', 'old_pkg', 'new_pkg')
        self.assertEqual(out, 'package://new_pkg/meshes/a.stl')
        out = replace_package_references(
            '$(find old_pkg)/xacro/a.xacro', 'old_pkg', 'new_pkg')
        self.assertEqual(out, '$(find new_pkg)/xacro/a.xacro')

    def test_replace_package_references_is_precise(self):
        # other package names containing the old name as a substring, and
        # plain-text occurrences, must be left untouched
        content = ('x="package://not_old_pkg/m.stl" '
                   'y="package://old_pkg/m.stl" old_pkg')
        out = replace_package_references(content, 'old_pkg', 'new_pkg')
        self.assertIn('package://not_old_pkg/m.stl', out)
        self.assertIn('package://new_pkg/m.stl', out)
        self.assertTrue(out.endswith(' old_pkg'))


if __name__ == '__main__':
    unittest.main()
