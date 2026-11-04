import math
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from skrobot.urdf import prepare_urdf_for_gazebo
from skrobot.urdf import prepare_urdf_for_gazebo_file


# Massless root, massless child with a box collision, a child that already
# has an inertial, a child whose inertial has zero mass, and a mimic joint.
_URDF = """<?xml version="1.0"?>
<robot name="t">
  <link name="base"/>
  <joint name="j1" type="revolute">
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="base"/><child link="arm"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <link name="arm">
    <visual><geometry><box size="0.1 0.2 0.3"/></geometry></visual>
    <collision><origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry><box size="0.1 0.2 0.3"/></geometry></collision>
  </link>
  <joint name="j2" type="revolute">
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <parent link="arm"/><child link="heavy"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <link name="heavy">
    <inertial><origin xyz="0.1 0 0" rpy="0 0 0"/><mass value="2.5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/></inertial>
    <visual><geometry><sphere radius="0.05"/></geometry></visual>
  </link>
  <joint name="j3" type="revolute">
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="heavy"/><child link="follower"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
    <mimic joint="j2" multiplier="-1" offset="0"/>
  </joint>
  <link name="follower">
    <inertial><mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>
    <visual><geometry><cylinder radius="0.02" length="0.1"/></geometry></visual>
  </link>
  <gazebo reference="arm"><mu1>0.5</mu1></gazebo>
</robot>
"""


def _root():
    return ET.fromstring(_URDF)


def _link(root, name):
    return root.find("link[@name='{}']".format(name))


def _mass(root, name):
    return float(_link(root, name).find('inertial/mass').get('value'))


def _inertia(root, name):
    tensor = _link(root, name).find('inertial/inertia')
    return [float(tensor.get(k)) for k in ('ixx', 'iyy', 'izz')]


class TestPrepareUrdfForGazebo(unittest.TestCase):

    def test_every_link_gets_a_positive_mass(self):
        root = _root()
        report = prepare_urdf_for_gazebo(root)
        for link in root.findall('link'):
            self.assertEqual(len(link.findall('inertial')), 1)
            self.assertGreater(_mass(root, link.get('name')), 0.0)
        self.assertEqual(sorted(report['added']),
                         ['arm', 'base', 'follower'])
        self.assertEqual(set(report['added'].values()), {'token'})
        self.assertEqual(report['replaced'], ['follower'])
        self.assertEqual(report['kept'], ['heavy'])
        self.assertAlmostEqual(_mass(root, 'base'), 0.01)
        np.testing.assert_allclose(_inertia(root, 'base'), [1e-6] * 3)

    def test_existing_inertial_untouched(self):
        root = _root()
        before = ET.tostring(_link(root, 'heavy').find('inertial'))
        prepare_urdf_for_gazebo(root, mass_from_geometry=True)
        after = ET.tostring(_link(root, 'heavy').find('inertial'))
        self.assertEqual(before, after)

    def test_token_values_configurable(self):
        root = _root()
        prepare_urdf_for_gazebo(root, token_mass=0.5, token_inertia=0.002)
        self.assertAlmostEqual(_mass(root, 'base'), 0.5)
        np.testing.assert_allclose(_inertia(root, 'base'), [0.002] * 3)
        with self.assertRaises(ValueError):
            prepare_urdf_for_gazebo(_root(), token_mass=0.0)

    def test_mass_from_geometry(self):
        root = _root()
        report = prepare_urdf_for_gazebo(root, mass_from_geometry=True,
                                         density=1000.0)
        # The collision box wins over the visual box, origin included.
        self.assertEqual(report['added']['arm'], 'collision:box')
        mass = 1000.0 * 0.1 * 0.2 * 0.3
        self.assertAlmostEqual(_mass(root, 'arm'), mass)
        com = _link(root, 'arm').find('inertial/origin').get('xyz')
        np.testing.assert_allclose([float(v) for v in com.split()],
                                   [0, 0, 0.5])
        np.testing.assert_allclose(
            _inertia(root, 'arm'),
            [mass / 12 * (0.2 ** 2 + 0.3 ** 2),
             mass / 12 * (0.1 ** 2 + 0.3 ** 2),
             mass / 12 * (0.1 ** 2 + 0.2 ** 2)])
        # No collision: the visual cylinder is used.
        self.assertEqual(report['added']['follower'], 'visual:cylinder')
        self.assertAlmostEqual(_mass(root, 'follower'),
                               1000.0 * math.pi * 0.02 ** 2 * 0.1)
        # No geometry at all: token.
        self.assertEqual(report['added']['base'], 'token')

    def test_strip_collision(self):
        root = _root()
        report = prepare_urdf_for_gazebo(root)
        self.assertEqual(report['collisions_removed'], 0)
        self.assertEqual(len(root.findall('link/collision')), 1)

        root = _root()
        report = prepare_urdf_for_gazebo(root, strip_collision=True,
                                         mass_from_geometry=True)
        self.assertEqual(report['collisions_removed'], 1)
        self.assertEqual(root.findall('link/collision'), [])
        # The collision was still used for the estimate before removal.
        self.assertEqual(report['added']['arm'], 'collision:box')

    def test_mimic_kept_or_stripped(self):
        root = _root()
        report = prepare_urdf_for_gazebo(root)
        self.assertEqual(report['mimic_joints'], {'j3': 'j2'})
        self.assertFalse(report['mimic_stripped'])
        self.assertIsNotNone(root.find("joint[@name='j3']/mimic"))

        root = _root()
        report = prepare_urdf_for_gazebo(root, strip_mimic=True)
        self.assertEqual(report['mimic_joints'], {'j3': 'j2'})
        self.assertTrue(report['mimic_stripped'])
        self.assertIsNone(root.find("joint[@name='j3']/mimic"))
        # The joint itself stays.
        self.assertEqual(root.find("joint[@name='j3']").get('type'),
                         'revolute')

    def test_mesh_geometry(self):
        import trimesh
        with tempfile.TemporaryDirectory() as tmp:
            trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(
                os.path.join(tmp, 'cube.stl'))
            root = ET.fromstring(
                '<robot name="m"><link name="a"><visual><geometry>'
                '<mesh filename="cube.stl" scale="0.1 0.2 0.3"/>'
                '</geometry></visual></link></robot>')
            report = prepare_urdf_for_gazebo(root, mass_from_geometry=True,
                                             density=1000.0, base_dir=tmp)
        self.assertEqual(report['added']['a'], 'visual:mesh')
        self.assertAlmostEqual(_mass(root, 'a'), 1000.0 * 0.1 * 0.2 * 0.3)

    def test_file_roundtrip_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'in.urdf')
            with open(src, 'w') as f:
                f.write(_URDF)
            out = os.path.join(tmp, 'out.urdf')
            report = prepare_urdf_for_gazebo_file(src, out)
            self.assertEqual(len(report['added']), 3)
            root = ET.parse(out).getroot()
            # Non-URDF-model tags survive the edit.
            self.assertIsNotNone(root.find("gazebo[@reference='arm']"))

            cli_out = os.path.join(tmp, 'cli.urdf')
            result = subprocess.run(
                [sys.executable, '-m', 'skrobot.apps.convert_urdf_for_gazebo',
                 src, '-o', cli_out, '--strip-collision', '--strip-mimic'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            root = ET.parse(cli_out).getroot()
            self.assertEqual(root.findall('link/collision'), [])
            self.assertIsNone(root.find('joint/mimic'))
            self.assertEqual(len(root.findall('link/inertial')), 4)


if __name__ == '__main__':
    unittest.main()
