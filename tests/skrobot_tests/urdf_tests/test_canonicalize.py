import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from skrobot.urdf import canonicalize_urdf
from skrobot.urdf import canonicalize_urdf_file


URDF_A = """<robot name="r">
  <joint name="j2" type="revolute">
    <child link="tip"/>
    <axis xyz="0 0 -1"/>
    <parent link="arm"/>
    <origin xyz="0.10000000001 0 -0" rpy="0 -0 0"/>
    <limit velocity="1" effort="10" lower="-1" upper="1"/>
  </joint>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="1 8.62e-16 0"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <link name="arm">
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </visual>
    <inertial>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <link name="base"/>
</robot>
"""

# Same robot: different element order, attribute order, number spelling.
URDF_B = """<robot name="r">
  <link name="base"/>
  <link name="arm">
    <inertial>
      <inertia izz="1" iyz="0" iyy="1" ixz="0" ixy="0" ixx="1"/>
      <mass value="1.0"/>
    </inertial>
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </visual>
  </link>
  <link name="tip"/>
  <joint type="revolute" name="j1">
    <origin rpy="-0 0 0" xyz="0 0 0.50000000004"/>
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="1 0 -0"/>
    <limit effort="10.0" lower="-1.0" upper="1.0" velocity="1.0"/>
  </joint>
  <joint type="revolute" name="j2">
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
    <parent link="arm"/>
    <child link="tip"/>
    <axis xyz="0 0 -1"/>
    <limit effort="10" lower="-1" upper="1" velocity="1"/>
  </joint>
</robot>
"""


class TestCanonicalizeUrdf(unittest.TestCase):

    def test_equivalent_documents_canonicalize_identically(self):
        self.assertEqual(canonicalize_urdf(URDF_A), canonicalize_urdf(URDF_B))

    def test_idempotent(self):
        once = canonicalize_urdf(URDF_A)
        self.assertEqual(once, canonicalize_urdf(once))

    def test_output_is_valid_urdf(self):
        root = ET.fromstring(canonicalize_urdf(URDF_A))
        self.assertEqual(root.tag, 'robot')
        self.assertEqual(len(root.findall('link')), 3)
        self.assertEqual(len(root.findall('joint')), 2)

    def test_negative_zero_and_noise_snapped(self):
        text = canonicalize_urdf(URDF_A)
        self.assertNotIn('-0 ', text)
        self.assertNotIn('"-0"', text)
        self.assertNotIn('8.62e-16', text)
        # axis "1 8.62e-16 0" -> "1 0 0"
        self.assertIn('<axis xyz="1 0 0" />', text)

    def test_links_and_joints_sorted_by_name(self):
        root = ET.fromstring(canonicalize_urdf(URDF_A))
        link_names = [e.get('name') for e in root.findall('link')]
        joint_names = [e.get('name') for e in root.findall('joint')]
        self.assertEqual(link_names, sorted(link_names))
        self.assertEqual(joint_names, sorted(joint_names))

    def test_kinematics_only_strips_geometry(self):
        text = canonicalize_urdf(URDF_A, kinematics_only=True)
        self.assertNotIn('<visual', text)
        self.assertNotIn('<inertial', text)
        self.assertIn('<axis', text)
        self.assertIn('<limit', text)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            canonicalize_urdf('not xml at all <')
        with self.assertRaises(ValueError):
            canonicalize_urdf('<not_a_robot/>')

    def test_file_roundtrip(self):
        tmpdir = tempfile.mkdtemp()
        in_path = os.path.join(tmpdir, 'in.urdf')
        out_path = os.path.join(tmpdir, 'out.urdf')
        with open(in_path, 'w', encoding='utf-8') as f:
            f.write(URDF_B)
        text = canonicalize_urdf_file(in_path, output_path=out_path)
        with open(out_path, encoding='utf-8') as f:
            self.assertEqual(f.read(), text)
        self.assertEqual(text, canonicalize_urdf(URDF_A))


if __name__ == '__main__':
    unittest.main()
