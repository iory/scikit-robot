"""Tests for skrobot.urdf.structure (link trees and validation)."""

import sys
import unittest

import pytest

from skrobot.data import kuka_urdfpath
from skrobot.urdf import kinematic_tree
from skrobot.urdf import print_urdf_tree
from skrobot.urdf import validate_urdf_structure
from skrobot.urdf import ValidationResult


VALID_URDF = """<robot name="sample">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="tip"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" velocity="1.0"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" velocity="1.0"/>
  </joint>
  <joint name="tip_joint" type="fixed">
    <parent link="link2"/>
    <child link="tip"/>
  </joint>
</robot>"""


@pytest.mark.skipif(
    sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
    reason="Skip in Python 2 and Python 3.6")
class TestPrintUrdfTree(unittest.TestCase):

    def test_tree_from_xml_string(self):
        tree = print_urdf_tree(VALID_URDF)
        self.assertIn('URDF Link Tree Structure:', tree)
        self.assertIn('base_link', tree)
        self.assertIn('tip', tree)

    def test_tree_from_file(self):
        tree = print_urdf_tree(kuka_urdfpath())
        self.assertIn('URDF Link Tree Structure:', tree)

    def test_tree_from_robot_model(self):
        from skrobot.models import Kuka
        tree = print_urdf_tree(Kuka())
        self.assertIn('URDF Link Tree Structure:', tree)


@pytest.mark.skipif(
    sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
    reason="Skip in Python 2 and Python 3.6")
class TestValidateUrdfStructure(unittest.TestCase):

    def test_valid_urdf(self):
        result = validate_urdf_structure(VALID_URDF)
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.summary['links_count'], 4)
        self.assertEqual(result.summary['joints_count'], 3)
        self.assertEqual(result.summary['base_links'], ['base_link'])
        self.assertEqual(result.summary['connected_components'], 1)

    def test_bad_link_reference(self):
        urdf = """<robot name="bad">
          <link name="a"/>
          <link name="b"/>
          <joint name="j" type="fixed">
            <parent link="missing"/>
            <child link="b"/>
          </joint>
        </robot>"""
        result = validate_urdf_structure(urdf)
        self.assertFalse(result.is_valid)
        self.assertTrue(
            any('non-existent parent' in e for e in result.errors))

    def test_disconnected_components(self):
        urdf = """<robot name="split">
          <link name="a"/>
          <link name="b"/>
          <link name="c"/>
          <joint name="j" type="fixed">
            <parent link="a"/>
            <child link="b"/>
          </joint>
        </robot>"""
        result = validate_urdf_structure(urdf)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.summary['connected_components'], 2)

    def test_zero_velocity_warning(self):
        urdf = """<robot name="zerovel">
          <link name="a"/>
          <link name="b"/>
          <joint name="j" type="revolute">
            <parent link="a"/>
            <child link="b"/>
            <axis xyz="0 0 1"/>
            <limit lower="-1" upper="1" velocity="0"/>
          </joint>
        </robot>"""
        result = validate_urdf_structure(urdf)
        self.assertTrue(any('zero velocity' in w for w in result.warnings))

    def test_str_renders_summary(self):
        result = validate_urdf_structure(VALID_URDF)
        text = str(result)
        self.assertIn('URDF Validation Summary:', text)
        self.assertIn('All validation checks passed!', text)


@pytest.mark.skipif(
    sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
    reason="Skip in Python 2 and Python 3.6")
class TestKinematicTree(unittest.TestCase):

    def test_annotations_and_collapse(self):
        tree = kinematic_tree(VALID_URDF)
        self.assertIn('Kinematic Tree', tree)
        # Movable joints are annotated with type and axis.
        self.assertIn('[revolute [0, 0, 1]]', tree)
        # The fixed tip frame is collapsed.
        self.assertIn('fixed frame', tree)
        self.assertIn('tip', tree)

    def test_branch_marker(self):
        urdf = """<robot name="branch">
          <link name="root"/>
          <link name="a"/>
          <link name="b"/>
          <joint name="j1" type="revolute">
            <parent link="root"/><child link="a"/>
            <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1"/>
          </joint>
          <joint name="j2" type="revolute">
            <parent link="root"/><child link="b"/>
            <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1"/>
          </joint>
        </robot>"""
        tree = kinematic_tree(urdf)
        self.assertIn('BRANCH(2)', tree)

    def test_no_collapse(self):
        tree = kinematic_tree(VALID_URDF, collapse_fixed=False)
        # Without collapsing, the fixed tip link is shown explicitly.
        self.assertIn('[fixed] tip', tree)

    def test_world_mode(self):
        # World mode loads a RobotModel to add world axes + link positions.
        tree = kinematic_tree(VALID_URDF, world=True)
        self.assertIn('world axes', tree)
        self.assertIn('@[', tree)        # link world position suffix

    def test_world_mode_from_robot_model(self):
        from skrobot.models import Kuka
        tree = kinematic_tree(Kuka(), world=True)
        self.assertIn('@[', tree)
