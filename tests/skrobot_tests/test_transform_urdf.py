import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from skrobot.urdf import transform_urdf_with_world_link


class TestTransformURDF(unittest.TestCase):

    def setUp(self):
        # Create a simple test URDF
        self.test_urdf = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </visual>
  </link>

  <link name="link1">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.5"/>
      </geometry>
    </visual>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>"""

    def test_find_root_link(self):
        """Test finding the root link."""
        from skrobot.urdf.modularize_urdf import find_root_link

        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(self.test_urdf)
            urdf_path = f.name

        try:
            root_link = find_root_link(urdf_path)
            self.assertEqual(root_link, 'base_link')
        finally:
            os.unlink(urdf_path)

    def test_transform_urdf_with_world_link_existing_name(self):
        """Test function with existing link name raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(self.test_urdf)
            urdf_path = f.name

        with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as output_f:
            output_path = output_f.name

        try:
            with self.assertRaises(ValueError):
                transform_urdf_with_world_link(
                    input_file=urdf_path,
                    output_file=output_path,
                    world_link_name='base_link'
                )
        finally:
            os.unlink(urdf_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_transform_urdf_with_world_link_function(self):
        """Test the convenience function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(self.test_urdf)
            urdf_path = f.name

        with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as output_f:
            output_path = output_f.name

        try:
            transform_urdf_with_world_link(
                input_file=urdf_path,
                output_file=output_path,
                x=0.5, y=1.0, z=1.5,
                roll=5.0, pitch=10.0, yaw=15.0,
                world_link_name='test_world'
            )

            self.assertTrue(os.path.exists(output_path))

            # Verify the output
            tree = ET.parse(output_path)
            root = tree.getroot()

            # Check that world link exists with correct name
            world_link = root.find("./link[@name='test_world']")
            self.assertIsNotNone(world_link)

            # Check joint
            joint = root.find("./joint[@name='test_world_to_base_link']")
            self.assertIsNotNone(joint)

            # Check origin
            origin = joint.find('origin')
            self.assertEqual(origin.get('xyz'), '0.5 1.0 1.5')

        finally:
            os.unlink(urdf_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_transform_urdf_with_world_link_nonexistent_file(self):
        """Test the convenience function with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            transform_urdf_with_world_link(
                input_file='nonexistent.urdf',
                output_file='output.urdf'
            )

    def test_transform_urdf_inplace_cli(self):
        """Test the CLI inplace functionality."""
        import subprocess

        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(self.test_urdf)
            urdf_path = f.name

        try:
            # Test --inplace option
            result = subprocess.run([
                'skr', 'transform-urdf', urdf_path,
                '--inplace', '--x', '1.5', '--yaw', '30'
            ], capture_output=True, text=True)

            self.assertEqual(result.returncode, 0)

            # Verify the file was modified in place
            self.assertTrue(os.path.exists(urdf_path))

            # Check the modified content
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            # Check that world link exists
            world_link = root.find("./link[@name='world']")
            self.assertIsNotNone(world_link)

            # Check joint with correct transform
            joint = root.find("./joint[@name='world_to_base_link']")
            self.assertIsNotNone(joint)

            origin = joint.find('origin')
            self.assertEqual(origin.get('xyz'), '1.5 0.0 0.0')
            # 30 degrees in radians ≈ 0.5236
            rpy_values = origin.get('rpy').split()
            self.assertAlmostEqual(float(rpy_values[2]), 0.5235987755982988, places=5)

        finally:
            if os.path.exists(urdf_path):
                os.unlink(urdf_path)
