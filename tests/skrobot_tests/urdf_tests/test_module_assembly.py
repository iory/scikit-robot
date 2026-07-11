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


class TestV2Design(unittest.TestCase):

    def test_port_frames_carry_translation(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = RobotModule.from_urdf('m', _write_module(tmp, 'm'))
        port = next(p for p in module.ports if p.name == 'dummy_link1')
        self.assertEqual(tuple(round(v, 9) for v in port.xyz),
                         (0.0, 0.0, 0.1))

    @unittest.skipUnless(shutil.which('zacro') is not None,
                         'zacro is not installed')
    def test_inline_and_xacro_engines_are_equivalent(self):
        import math

        def _joint_key(joint):
            origin = joint.find('origin')
            xyz = tuple(round(float(v), 9) for v in
                        (origin.get('xyz', '0 0 0').split()
                         if origin is not None else '0 0 0'.split()))
            rpy = tuple(round(float(v), 9) for v in
                        (origin.get('rpy', '0 0 0').split()
                         if origin is not None else '0 0 0'.split()))
            return (joint.get('name'), joint.get('type'),
                    joint.find('parent').get('link'),
                    joint.find('child').get('link'), xyz, rpy)

        with tempfile.TemporaryDirectory() as tmp:
            module_a = RobotModule.from_urdf('mod_a',
                                             _write_module(tmp, 'mod_a'))
            module_b = RobotModule.from_urdf('mod_b',
                                             _write_module(tmp, 'mod_b'))
            outputs = {}
            for engine in ('inline', 'xacro'):
                assembly = RobotAssembly('combo')
                assembly.add_module_instance('a1', module_a)
                assembly.add_module_instance('b1', module_b)
                assembly.connect('a1', 'dummy_link1', 'b1', 'base_link',
                                 x=0.1, y=0.2, z=0.3,
                                 roll=0.1, pitch=0.2, yaw=math.pi / 6)
                assembly.set_root('a1', 'base_link')
                path = os.path.join(tmp, f'{engine}.urdf')
                outputs[engine] = ET.parse(
                    assembly.build(output_path=path, engine=engine)).getroot()
        for tag, key in (('link', lambda el: el.get('name')),
                         ('joint', _joint_key)):
            inline_set = {key(el) for el in outputs['inline'].findall(tag)}
            xacro_set = {key(el) for el in outputs['xacro'].findall(tag)}
            self.assertEqual(inline_set, xacro_set)

    def test_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            modules = [RobotModule.from_urdf(f'm{i}',
                                             _write_module(tmp, f'm{i}'))
                       for i in range(3)]
            assembly = RobotAssembly('loop')
            for i, module in enumerate(modules):
                assembly.add_module_instance(f'i{i}', module)
            assembly.connect('i0', 'dummy_link1', 'i1', 'base_link')
            assembly.connect('i1', 'dummy_link1', 'i2', 'base_link')
            assembly.connect('i2', 'dummy_link1', 'i0', 'base_link')
            with self.assertRaises(ValueError):
                assembly.build(output_path=os.path.join(tmp, 'x.urdf'))

    def test_from_dict_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            module_a = RobotModule.from_urdf('mod_a',
                                             _write_module(tmp, 'mod_a'))
            module_b = RobotModule.from_urdf('mod_b',
                                             _write_module(tmp, 'mod_b'))
            assembly = RobotAssembly('combo')
            assembly.add_module_instance('a1', module_a)
            assembly.add_module_instance('b1', module_b)
            assembly.connect('a1', 'dummy_link1', 'b1', 'base_link',
                             x=0.1, yaw=0.2)
            assembly.set_root('a1', 'base_link')
            data = assembly.to_dict()
            rebuilt = RobotAssembly.from_dict(data)
            self.assertEqual(rebuilt.to_dict(), data)
        self.assertEqual(rebuilt.root_instance, 'a1')

    def test_mate_places_child_port_onto_parent_port(self):
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
            assembly.connect('a1', 'dummy_link1', 'b1', 'dummy_link1',
                             mate=True)
            conn = assembly.connections[0]
            port = next(p for p in module_b.ports
                        if p.name == 'dummy_link1')
        transform = xyzrpy2matrix(conn.xyz, conn.rpy)
        # the child port frame, placed by the connection transform, must sit
        # at the parent port origin with its Z axis opposed to the parent Z
        placed_origin = transform @ np.append(np.asarray(port.xyz), 1.0)
        np.testing.assert_allclose(placed_origin[:3], np.zeros(3),
                                   atol=1e-12)
        placed_z = transform[:3, :3] @ np.asarray(port.z_axis)
        np.testing.assert_allclose(placed_z, [0.0, 0.0, -1.0], atol=1e-12)

    def test_mate_rejects_explicit_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            module_a = RobotModule.from_urdf('mod_a',
                                             _write_module(tmp, 'mod_a'))
            module_b = RobotModule.from_urdf('mod_b',
                                             _write_module(tmp, 'mod_b'))
            assembly = RobotAssembly('combo')
            assembly.add_module_instance('a1', module_a)
            assembly.add_module_instance('b1', module_b)
            with self.assertRaises(ValueError):
                assembly.connect('a1', 'dummy_link1', 'b1', 'dummy_link1',
                                 x=0.5, mate=True)
