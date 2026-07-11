import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from skrobot.urdf import RobotAssembly
from skrobot.urdf import RobotModule


_MODULE_URDF = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link"/>
  <link name="dummy_link1"/>
  <joint name="j1" type="revolute">
    <parent link="base_link"/>
    <child link="dummy_link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _write_module(directory, name):
    path = os.path.join(directory, name + '.urdf')
    with open(path, 'w') as f:
        f.write(_MODULE_URDF.format(name=name))
    return path


class TestRobotModule(unittest.TestCase):

    def test_from_urdf_extracts_ports_and_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = RobotModule.from_urdf(
                'hinge', _write_module(tmp, 'hinge'))
        self.assertEqual(module.root_link, 'base_link')
        self.assertEqual(sorted(module.get_port_names()),
                         ['base_link', 'dummy_link1'])

    def test_from_urdf_string(self):
        module = RobotModule.from_urdf_string(
            'hinge', _MODULE_URDF.format(name='hinge'))
        self.assertEqual(module.root_link, 'base_link')


class TestRobotAssembly(unittest.TestCase):

    def _make_assembly(self, tmp):
        module_a = RobotModule.from_urdf('mod_a', _write_module(tmp, 'mod_a'))
        module_b = RobotModule.from_urdf('mod_b', _write_module(tmp, 'mod_b'))
        assembly = RobotAssembly('combo')
        assembly.add_module_instance('a1', module_a)
        assembly.add_module_instance('b1', module_b)
        assembly.connect('a1', 'dummy_link1', 'b1', 'base_link')
        assembly.set_root('a1', 'base_link')
        return assembly

    def test_connect_and_to_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._make_assembly(tmp)
            data = assembly.to_dict()
        self.assertEqual(sorted(data['instances']), ['a1', 'b1'])
        self.assertEqual(len(data['connections']), 1)

    def test_to_dict_keeps_connection_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            module_a = RobotModule.from_urdf('mod_a',
                                             _write_module(tmp, 'mod_a'))
            module_b = RobotModule.from_urdf('mod_b',
                                             _write_module(tmp, 'mod_b'))
            assembly = RobotAssembly('combo')
            assembly.add_module_instance('a1', module_a)
            assembly.add_module_instance('b1', module_b)
            assembly.connect('a1', 'dummy_link1', 'b1', 'base_link',
                             x=0.1, y=0.2, z=0.3,
                             roll=0.4, pitch=0.5, yaw=0.6)
            conn = assembly.to_dict()['connections'][0]
        self.assertEqual((conn['x'], conn['y'], conn['z']), (0.1, 0.2, 0.3))
        self.assertEqual((conn['roll'], conn['pitch'], conn['yaw']),
                         (0.4, 0.5, 0.6))

    def test_reverse_adjacency_is_rigid_inverse(self):
        import numpy as np

        from skrobot.coordinates.math import xyzrpy2matrix
        with tempfile.TemporaryDirectory() as tmp:
            module_a = RobotModule.from_urdf('mod_a',
                                             _write_module(tmp, 'mod_a'))
            module_b = RobotModule.from_urdf('mod_b',
                                             _write_module(tmp, 'mod_b'))
            assembly = RobotAssembly('combo')
            assembly.add_module_instance('a1', module_a)
            assembly.add_module_instance('b1', module_b)
            # multi-axis rotation: a component-wise negation is NOT the
            # inverse here, only a rigid inverse composes to identity
            assembly.connect('a1', 'dummy_link1', 'b1', 'base_link',
                             x=0.1, y=0.2, z=0.3,
                             roll=0.4, pitch=0.5, yaw=0.6)
            adj = assembly.get_adjacency_list()
        fwd = next(e for e in adj['a1'] if e[0] == 'b1')
        rev = next(e for e in adj['b1'] if e[0] == 'a1')
        forward = xyzrpy2matrix(fwd[3], fwd[4])
        reverse = xyzrpy2matrix(rev[3], rev[4])
        np.testing.assert_allclose(forward @ reverse, np.eye(4), atol=1e-12)

    @unittest.skipUnless(shutil.which('zacro') is not None,
                         'zacro is not installed')
    def test_non_root_child_port_still_attaches_via_root(self):
        # child_port records which port the connection was declared with;
        # the module nevertheless attaches through its root link (the
        # documented convention -- callers use child_port for placement).
        with tempfile.TemporaryDirectory() as tmp:
            module_a = RobotModule.from_urdf('mod_a',
                                             _write_module(tmp, 'mod_a'))
            module_b = RobotModule.from_urdf('mod_b',
                                             _write_module(tmp, 'mod_b'))
            assembly = RobotAssembly('combo')
            assembly.add_module_instance('a1', module_a)
            assembly.add_module_instance('b1', module_b)
            assembly.connect('a1', 'dummy_link1', 'b1', 'dummy_link1')
            assembly.set_root('a1', 'base_link')
            urdf_path = assembly.build(
                output_path=os.path.join(tmp, 'x.urdf'))
            root = ET.parse(urdf_path).getroot()
        fixed = [j for j in root.findall('joint')
                 if j.get('type') == 'fixed'
                 and 'b1' in j.find('child').get('link')]
        self.assertTrue(fixed)
        # attachment goes to b1's ROOT link, not the declared dummy port
        self.assertTrue(
            fixed[0].find('child').get('link').endswith('base_link'),
            fixed[0].find('child').get('link'))

    @unittest.skipUnless(shutil.which('zacro') is not None,
                         'zacro is not installed')
    def test_build_produces_connected_urdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._make_assembly(tmp)
            urdf_path = assembly.build(
                output_path=os.path.join(tmp, 'combo.urdf'))
            root = ET.parse(urdf_path).getroot()
        link_names = {link.get('name') for link in root.findall('link')}
        # both modules present with their instance prefixes
        self.assertTrue(any('a1' in n for n in link_names), link_names)
        self.assertTrue(any('b1' in n for n in link_names), link_names)
        # exactly one link never appears as a child -> a single tree
        child_links = {j.find('child').get('link')
                       for j in root.findall('joint')}
        roots = link_names - child_links
        self.assertEqual(len(roots), 1, roots)
        # the connection is a fixed joint from a1's port to b1's root
        fixed = [j for j in root.findall('joint')
                 if j.get('type') == 'fixed'
                 and 'a1' in j.find('parent').get('link')
                 and 'b1' in j.find('child').get('link')]
        self.assertTrue(fixed)


if __name__ == '__main__':
    unittest.main()
