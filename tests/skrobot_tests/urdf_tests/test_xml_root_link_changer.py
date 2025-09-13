"""Test for XML-based URDF root link changer."""

import os
import sys
import tempfile
import unittest

import pytest

from skrobot.data import kuka_urdfpath
from skrobot.urdf import URDFXMLRootLinkChanger


@pytest.mark.skipif(
    sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
    reason="Skip in Python 2 and Python 3.6")
class TestURDFXMLRootLinkChanger(unittest.TestCase):

    def setUp(self):
        self.urdf_path = kuka_urdfpath()
        self.changer = URDFXMLRootLinkChanger(self.urdf_path)

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init_with_valid_urdf_path(self):
        changer = URDFXMLRootLinkChanger(self.urdf_path)
        self.assertIsNotNone(changer.tree)
        self.assertIsNotNone(changer.root)
        self.assertIsInstance(changer.links, dict)
        self.assertIsInstance(changer.joints, dict)
        self.assertIsInstance(changer.joint_tree, dict)

    def test_init_with_invalid_urdf_path(self):
        with self.assertRaises(FileNotFoundError):
            URDFXMLRootLinkChanger("non_existent_file.urdf")

    def test_list_links(self):
        links = self.changer.list_links()
        self.assertIsInstance(links, list)
        self.assertGreater(len(links), 0)

        for link in links:
            self.assertIsInstance(link, str)

    def test_get_current_root_link(self):
        root_link = self.changer.get_current_root_link()
        self.assertIsInstance(root_link, str)
        self.assertIn(root_link, self.changer.list_links())

    def test_change_root_link_invalid_name(self):
        output_path = os.path.join(self.temp_dir, "test_invalid.urdf")
        with self.assertRaises(ValueError):
            self.changer.change_root_link("non_existent_link", output_path)

    def test_change_root_link_valid_name(self):
        links = self.changer.list_links()
        if len(links) > 1:
            current_root = self.changer.get_current_root_link()
            new_root_candidates = [link for link in links
                                   if link != current_root]

            if new_root_candidates:
                new_root = new_root_candidates[0]
                output_path = os.path.join(self.temp_dir, "test_changed.urdf")

                self.changer.change_root_link(new_root, output_path)

                self.assertTrue(os.path.exists(output_path))

                new_changer = URDFXMLRootLinkChanger(output_path)
                actual_new_root = new_changer.get_current_root_link()
                self.assertEqual(actual_new_root, new_root)

    def test_change_root_link_same_as_current(self):
        current_root = self.changer.get_current_root_link()
        output_path = os.path.join(self.temp_dir, "test_same.urdf")

        self.changer.change_root_link(current_root, output_path)

        self.assertTrue(os.path.exists(output_path))

        new_changer = URDFXMLRootLinkChanger(output_path)
        actual_root = new_changer.get_current_root_link()
        self.assertEqual(actual_root, current_root)

    def test_find_path_to_link(self):
        current_root = self.changer.get_current_root_link()
        links = self.changer.list_links()

        if len(links) > 1:
            target_link = None
            for link in links:
                if link != current_root:
                    target_link = link
                    break

            if target_link:
                path = self.changer._find_path_to_link(
                    current_root, target_link)
                self.assertIsInstance(path, list)
                # Path should contain tuples of (parent, child, joint)
                for item in path:
                    self.assertIsInstance(item, tuple)
                    self.assertEqual(len(item), 3)

    def test_change_root_link_with_fixed_joints(self):
        """Test root link changing with fixed joints (no axis elements)."""
        fixed_joint_urdf_content = '''<?xml version="1.0"?>
<robot name="test_robot">
  <link name="world"/>
  <link name="base_link"/>
  <link name="end_link"/>

  <joint name="world_to_base" type="fixed">
    <parent link="world"/>
    <child link="base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <joint name="base_to_end" type="revolute">
    <parent link="base_link"/>
    <child link="end_link"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>'''

        test_urdf_path = os.path.join(self.temp_dir, "fixed_joint_test.urdf")
        with open(test_urdf_path, 'w') as f:
            f.write(fixed_joint_urdf_content)

        changer = URDFXMLRootLinkChanger(test_urdf_path)

        output_path = os.path.join(self.temp_dir, "fixed_joint_changed.urdf")
        changer.change_root_link("base_link", output_path)

        self.assertTrue(os.path.exists(output_path))

        new_changer = URDFXMLRootLinkChanger(output_path)
        actual_new_root = new_changer.get_current_root_link()
        self.assertEqual(actual_new_root, "base_link")

    def test_change_root_link_with_mimic_joints(self):
        """Test root link changing with mimic joints and naming conflicts."""
        mimic_joint_urdf_content = '''<?xml version="1.0"?>
<robot name="test_robot">
  <link name="world"/>
  <link name="base_link"/>
  <link name="joint1"/>
  <link name="joint2"/>

  <joint name="world_to_base_joint" type="fixed">
    <parent link="world"/>
    <child link="base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="joint1"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>

  <joint name="joint2_joint" type="revolute">
    <parent link="joint1"/>
    <child link="joint2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
    <mimic joint="joint1" multiplier="2.0"/>
  </joint>
</robot>'''

        test_urdf_path = os.path.join(self.temp_dir, "mimic_joint_test.urdf")
        with open(test_urdf_path, 'w') as f:
            f.write(mimic_joint_urdf_content)

        changer = URDFXMLRootLinkChanger(test_urdf_path)

        output_path = os.path.join(self.temp_dir, "mimic_joint_changed.urdf")
        changer.change_root_link("base_link", output_path)

        self.assertTrue(os.path.exists(output_path))

        new_changer = URDFXMLRootLinkChanger(output_path)
        actual_new_root = new_changer.get_current_root_link()
        self.assertEqual(actual_new_root, "base_link")

    def test_none_element_handling(self):
        """Test handling of None elements during XML parsing."""
        malformed_urdf_content = '''<?xml version="1.0"?>
<robot name="test_robot">
  <link name="world"/>
  <link name="base_link"/>

  <joint name="world_to_base" type="fixed">
    <parent link="world"/>
    <child link="base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
</robot>'''

        test_urdf_path = os.path.join(self.temp_dir, "malformed_test.urdf")
        with open(test_urdf_path, 'w') as f:
            f.write(malformed_urdf_content)

        changer = URDFXMLRootLinkChanger(test_urdf_path)

        output_path = os.path.join(self.temp_dir, "malformed_changed.urdf")

        try:
            changer.change_root_link("base_link", output_path)
            self.assertTrue(os.path.exists(output_path))
        except AttributeError as e:
            if "'NoneType' object has no attribute 'get'" in str(e):
                self.fail("NoneType error not properly handled: {}".format(e))
            else:
                raise

    def test_complex_kinematic_tree_root_change(self):
        """Test root link change in complex kinematic trees with multiple branches."""
        complex_urdf_content = '''<?xml version="1.0"?>
<robot name="complex_robot">
  <link name="world"/>
  <link name="base_link"/>
  <link name="arm1_link"/>
  <link name="arm2_link"/>
  <link name="gripper_base"/>
  <link name="gripper_finger1"/>
  <link name="gripper_finger2"/>

  <joint name="world_to_base_joint" type="fixed">
    <parent link="world"/>
    <child link="base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <joint name="base_to_arm1_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm1_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>

  <joint name="arm1_to_arm2_joint" type="revolute">
    <parent link="arm1_link"/>
    <child link="arm2_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>

  <joint name="arm2_to_gripper_joint" type="fixed">
    <parent link="arm2_link"/>
    <child link="gripper_base"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <joint name="gripper_finger1_joint" type="revolute">
    <parent link="gripper_base"/>
    <child link="gripper_finger1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.5" effort="5" velocity="0.5"/>
  </joint>

  <joint name="gripper_finger2_joint" type="revolute">
    <parent link="gripper_base"/>
    <child link="gripper_finger2"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.5" effort="5" velocity="0.5"/>
    <mimic joint="gripper_finger1_joint" multiplier="-1.0"/>
  </joint>
</robot>'''

        test_urdf_path = os.path.join(self.temp_dir, "complex_test.urdf")
        with open(test_urdf_path, 'w') as f:
            f.write(complex_urdf_content)

        changer = URDFXMLRootLinkChanger(test_urdf_path)

        output_path = os.path.join(self.temp_dir, "complex_changed.urdf")
        changer.change_root_link("gripper_base", output_path)

        self.assertTrue(os.path.exists(output_path))

        new_changer = URDFXMLRootLinkChanger(output_path)
        actual_new_root = new_changer.get_current_root_link()
        self.assertEqual(actual_new_root, "gripper_base")

        new_links = new_changer.list_links()
        self.assertIn("gripper_finger1", new_links)
        self.assertIn("gripper_finger2", new_links)


if __name__ == '__main__':
    unittest.main()
