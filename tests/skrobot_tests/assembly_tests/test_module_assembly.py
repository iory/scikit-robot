import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from skrobot.assembly import RobotAssembly
from skrobot.assembly import RobotModule


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
  <transmission name="tr1">
    <joint name="j1"><hardwareInterface>EffortJointInterface</hardwareInterface></joint>
    <actuator name="motor1"><mechanicalReduction>1</mechanicalReduction></actuator>
  </transmission>
  <gazebo reference="dummy_link1"><selfCollide>false</selfCollide></gazebo>
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


class TestV3Design(unittest.TestCase):

    def _two_module_assembly(self, tmp, **connect_kwargs):
        module_a = RobotModule.from_urdf('mod_a', _write_module(tmp, 'mod_a'))
        module_b = RobotModule.from_urdf('mod_b', _write_module(tmp, 'mod_b'))
        assembly = RobotAssembly('combo')
        assembly.add_module_instance('a1', module_a)
        assembly.add_module_instance('b1', module_b)
        assembly.connect('a1', 'dummy_link1', 'b1', 'dummy_link1',
                         **connect_kwargs)
        assembly.set_root('a1', 'base_link')
        return assembly

    def test_attach_port_reroots_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._two_module_assembly(tmp, attach='port')
            root = ET.parse(assembly.build(
                output_path=os.path.join(tmp, 'x.urdf'))).getroot()
        joints = root.findall('joint')
        # the connector now attaches b1 through its DECLARED port
        connector = [j for j in joints if j.get('type') == 'fixed'
                     and j.find('child').get('link') == 'b1_dummy_link1']
        self.assertTrue(connector)
        # the child chain was re-rooted: b1_base_link is now DOWNSTREAM of
        # b1_dummy_link1 (it appears as a child link of some joint)
        child_links = {j.find('child').get('link') for j in joints}
        self.assertIn('b1_base_link', child_links)
        # still a single tree rooted at world
        link_names = {link.get('name') for link in root.findall('link')}
        roots = link_names - child_links
        self.assertEqual(roots, {'world'})

    def test_attach_port_with_mate_is_pure_flip(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._two_module_assembly(tmp, attach='port',
                                                 mate=True)
            conn = assembly.connections[0]
        # re-rooted child: the port frame IS the child root frame, so the
        # mate transform is a pure 180-degree flip with no translation
        np.testing.assert_allclose(conn.xyz, np.zeros(3), atol=1e-12)
        from skrobot.coordinates.math import rpy2matrix
        rotation = rpy2matrix(*conn.rpy)
        np.testing.assert_allclose(rotation @ [0, 0, 1], [0, 0, -1],
                                   atol=1e-12)

    def test_attach_round_trips_through_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._two_module_assembly(tmp, attach='port')
            data = assembly.to_dict()
            self.assertEqual(data['connections'][0]['attach'], 'port')
            rebuilt = RobotAssembly.from_dict(data)
            self.assertEqual(rebuilt.to_dict(), data)

    def test_build_robot_model_directly(self):
        from skrobot.model import RobotModel
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._two_module_assembly(tmp)
            robot = assembly.build_robot_model()
            # compare against loading the built URDF from disk
            urdf_path = assembly.build(
                output_path=os.path.join(tmp, 'x.urdf'))
            disk_links = {link.get('name') for link in
                          ET.parse(urdf_path).getroot().findall('link')}
        self.assertIsInstance(robot, RobotModel)
        model_links = {link.name for link in robot.link_list}
        self.assertEqual(model_links, disk_links)
        # prefixed joints exist and are actuable
        joint_names = {j.name for j in robot.joint_list}
        self.assertIn('a1_j1', joint_names)
        self.assertIn('b1_j1', joint_names)


class TestV4PortsAndMating(unittest.TestCase):

    _ROTATED_MODULE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link"/>
  <link name="dummy_link1"/>
  <joint name="j1" type="fixed">
    <parent link="base_link"/>
    <child link="dummy_link1"/>
    <origin xyz="0.1 0.05 0.2" rpy="0.3 -0.2 0.5"/>
  </joint>
</robot>
"""

    def _rotated_module(self, tmp, name):
        path = os.path.join(tmp, name + '.urdf')
        with open(path, 'w') as f:
            f.write(self._ROTATED_MODULE.format(name=name))
        return RobotModule.from_urdf(name, path)

    def test_ports_carry_full_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = self._rotated_module(tmp, 'm')
        port = next(p for p in module.ports if p.name == 'dummy_link1')
        self.assertEqual(tuple(round(v, 9) for v in port.rpy),
                         (0.3, -0.2, 0.5))
        self.assertEqual(tuple(round(v, 9) for v in port.xyz),
                         (0.1, 0.05, 0.2))

    def test_keyed_mate_is_fully_determined(self):
        import numpy as np

        from skrobot.coordinates.math import rpy2homogeneous
        from skrobot.coordinates.math import xyzrpy2matrix
        with tempfile.TemporaryDirectory() as tmp:
            module_a = self._rotated_module(tmp, 'a')
            module_b = self._rotated_module(tmp, 'b')
            assembly = RobotAssembly('combo')
            assembly.add_module_instance('a1', module_a)
            assembly.add_module_instance('b1', module_b)
            assembly.connect('a1', 'dummy_link1', 'b1', 'dummy_link1',
                             mate=True, yaw=0.4)
            conn = assembly.connections[0]
            port = next(p for p in module_b.ports
                        if p.name == 'dummy_link1')
        # the seated child-port frame must be EXACTLY Rz(yaw) @ Rx(pi) in
        # parent-port coordinates: origins coincide, Z opposed, X keyed
        transform = xyzrpy2matrix(conn.xyz, conn.rpy)
        seated = transform @ xyzrpy2matrix(port.xyz, port.rpy)
        expected = rpy2homogeneous(np.pi, 0.0, 0.4)
        np.testing.assert_allclose(seated, expected, atol=1e-12)

    def test_unknown_port_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            module_a = self._rotated_module(tmp, 'a')
            module_b = self._rotated_module(tmp, 'b')
            assembly = RobotAssembly('combo')
            assembly.add_module_instance('a1', module_a)
            assembly.add_module_instance('b1', module_b)
            with self.assertRaises(ValueError):
                assembly.connect('a1', 'no_such_port', 'b1', 'base_link')

    def test_port_type_compatibility_enforced(self):
        from skrobot.assembly import Port
        with tempfile.TemporaryDirectory() as tmp:
            module_a = self._rotated_module(tmp, 'a')
            module_b = self._rotated_module(tmp, 'b')
        # curate the catalogs: both ports become outputs
        for module in (module_a, module_b):
            module.ports = [Port(name='dummy_link1', port_type='output'),
                            Port(name='base_link', port_type='input')]
        assembly = RobotAssembly('combo')
        assembly.add_module_instance('a1', module_a)
        assembly.add_module_instance('b1', module_b)
        with self.assertRaises(ValueError):
            assembly.connect('a1', 'dummy_link1', 'b1', 'dummy_link1')
        # output -> input is fine
        assembly.connect('a1', 'dummy_link1', 'b1', 'base_link')

    def test_compatible_types_whitelist(self):
        from skrobot.assembly import Port
        with tempfile.TemporaryDirectory() as tmp:
            module_a = self._rotated_module(tmp, 'a')
            module_b = self._rotated_module(tmp, 'b')
        module_a.ports = [
            Port(name='dummy_link1', port_type='output',
                 compatible_types=['servo_flange'])]
        module_b.ports = [
            Port(name='base_link', port_type='input'),
            Port(name='dummy_link1', port_type='servo_flange')]
        assembly = RobotAssembly('combo')
        assembly.add_module_instance('a1', module_a)
        assembly.add_module_instance('b1', module_b)
        with self.assertRaises(ValueError):
            assembly.connect('a1', 'dummy_link1', 'b1', 'base_link')
        assembly.connect('a1', 'dummy_link1', 'b1', 'dummy_link1')


_GROUND_URDF = """<?xml version="1.0"?>
<robot name="ground">
  <link name="base_link"/>
  <link name="g1"/>
  <link name="g2"/>
  <joint name="fg1" type="fixed">
    <parent link="base_link"/><child link="g1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="fg2" type="fixed">
    <parent link="base_link"/><child link="g2"/>
    <origin xyz="{g2_xyz}" rpy="0 0 0"/>
  </joint>
</robot>
"""

_GROUND3_URDF = """<?xml version="1.0"?>
<robot name="ground3">
  <link name="base_link"/>
  <link name="g1"/>
  <link name="g2"/>
  <link name="g3"/>
  <joint name="fg1" type="fixed">
    <parent link="base_link"/><child link="g1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="fg2" type="fixed">
    <parent link="base_link"/><child link="g2"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="fg3" type="fixed">
    <parent link="base_link"/><child link="g3"/>
    <origin xyz="0.4 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""

_BAR_URDF = """<?xml version="1.0"?>
<robot name="bar">
  <link name="base_link"/>
  <link name="arm"/>
  <link name="tip"/>
  <joint name="hinge" type="revolute">
    <parent link="base_link"/><child link="arm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
  </joint>
  <joint name="tipj" type="fixed">
    <parent link="arm"/><child link="tip"/>
    <origin xyz="{tip_xyz}" rpy="{tip_rpy}"/>
  </joint>
</robot>
"""


class TestV5LoopClosures(unittest.TestCase):
    """connect(loop=True): cut edges, the relay sidecar and exact mimic."""

    @staticmethod
    def _write(tmp, name, content):
        path = os.path.join(tmp, name + '.urdf')
        with open(path, 'w') as f:
            f.write(content)
        return RobotModule.from_urdf(name, path)

    def _four_bar(self, tmp, g2_xyz='0.2 0 0', crank2_tip='0 0.1 0',
                  coupler_tip_rpy='0 0 0'):
        """Ground + two cranks + coupler, loop-closed at the coupler tip.

        The defaults make the hinge quadrilateral a rectangle (a
        parallelogram); pass a shifted ``g2_xyz``/``crank2_tip`` pair that
        still closes at the zero pose for a non-parallelogram four-bar.
        """
        ground = self._write(tmp, 'ground',
                             _GROUND_URDF.format(g2_xyz=g2_xyz))
        crank = self._write(tmp, 'crank',
                            _BAR_URDF.format(tip_xyz='0 0.1 0',
                                             tip_rpy='0 0 0'))
        crank2 = self._write(tmp, 'crank2',
                             _BAR_URDF.format(tip_xyz=crank2_tip,
                                              tip_rpy='0 0 0'))
        coupler = self._write(tmp, 'coupler',
                              _BAR_URDF.format(tip_xyz='0.2 0 0',
                                               tip_rpy=coupler_tip_rpy))
        assembly = RobotAssembly('fourbar')
        assembly.add_module_instance('g', ground)
        assembly.add_module_instance('c1', crank)
        assembly.add_module_instance('c2', crank2)
        assembly.add_module_instance('cp', coupler)
        assembly.connect('g', 'g1', 'c1', 'base_link')
        assembly.connect('g', 'g2', 'c2', 'base_link')
        assembly.connect('c1', 'tip', 'cp', 'base_link')
        assembly.connect('cp', 'tip', 'c2', 'tip', loop=True)
        assembly.set_root('g', 'base_link')
        return assembly

    def test_loop_rejects_transform_mate_and_attach(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._four_bar(tmp)
            for kwargs in ({'yaw': 0.5}, {'x': 0.1}, {'mate': True},
                           {'attach': 'port'}):
                with self.assertRaises(ValueError):
                    assembly.connect('c1', 'arm', 'c2', 'arm',
                                     loop=True, **kwargs)

    def test_loop_builds_and_writes_relay_sidecar(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._four_bar(tmp)
            path = assembly.build(
                output_path=os.path.join(tmp, 'fourbar.urdf'))
            sidecar = os.path.join(os.path.dirname(path),
                                   'loop_closures.yaml')
            self.assertTrue(os.path.exists(sidecar))
            with open(sidecar) as f:
                config = yaml.safe_load(f)
        self.assertEqual(config['independent'], ['c1_hinge'])
        self.assertEqual(config['dependent'], ['c2_hinge', 'cp_hinge'])
        closure = config['closures'][0]
        self.assertEqual(closure['link_a'], 'cp_tip')
        self.assertEqual(closure['link_b'], 'c2_tip')
        for got, want in zip(closure['point'], [0.2, 0.1, 0.0]):
            self.assertAlmostEqual(got, want, places=8)
        for got, want in zip(closure['axis'], [0.0, 0.0, 1.0]):
            self.assertAlmostEqual(got, want, places=8)

    def test_parallelogram_gets_exact_mimic(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._four_bar(tmp)
            root = ET.parse(assembly.build(
                output_path=os.path.join(tmp, 'fourbar.urdf'))).getroot()
        mimics = {j.get('name'): j.find('mimic')
                  for j in root.findall('joint')
                  if j.find('mimic') is not None}
        self.assertEqual(sorted(mimics), ['c2_hinge', 'cp_hinge'])
        for name, multiplier in (('c2_hinge', 1.0), ('cp_hinge', -1.0)):
            self.assertEqual(mimics[name].get('joint'), 'c1_hinge')
            self.assertAlmostEqual(
                float(mimics[name].get('multiplier')), multiplier)

    def test_general_four_bar_gets_no_mimic(self):
        # crank2 is shorter and its ground pivot shifted so the loop still
        # closes at zero but the quadrilateral is not a parallelogram
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._four_bar(tmp, g2_xyz='0.2 0.05 0',
                                      crank2_tip='0 0.05 0')
            root = ET.parse(assembly.build(
                output_path=os.path.join(tmp, 'fourbar.urdf'))).getroot()
            sidecar = os.path.join(tmp, 'loop_closures.yaml')
            self.assertTrue(os.path.exists(sidecar))
        self.assertEqual([j.get('name') for j in root.findall('joint')
                          if j.find('mimic') is not None], [])

    def test_z_opposed_loop_ports_still_close(self):
        # the keyed-mate convention seats ports Z-opposed; the hinge is a
        # LINE, so an anti-parallel port Z must work like an aligned one
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._four_bar(
                tmp, coupler_tip_rpy='3.141592653589793 0 0')
            root = ET.parse(assembly.build(
                output_path=os.path.join(tmp, 'fourbar.urdf'))).getroot()
        mimics = {j.get('name'): j.find('mimic')
                  for j in root.findall('joint')
                  if j.find('mimic') is not None}
        self.assertEqual(sorted(mimics), ['c2_hinge', 'cp_hinge'])
        for name, multiplier in (('c2_hinge', 1.0), ('cp_hinge', -1.0)):
            self.assertEqual(mimics[name].get('joint'), 'c1_hinge')
            self.assertAlmostEqual(
                float(mimics[name].get('multiplier')), multiplier)

    def test_unrelated_movable_joint_stays_independent(self):
        # relay contract: independent = every movable joint the relay must
        # take from the incoming joint states, loop-related or not
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._four_bar(tmp)
            extra = self._write(tmp, 'extra',
                                _BAR_URDF.format(tip_xyz='0 0.1 0',
                                                 tip_rpy='0 0 0'))
            assembly.add_module_instance('x1', extra)
            assembly.connect('g', 'g2', 'x1', 'base_link', x=1.0)
            assembly.build(output_path=os.path.join(tmp, 'fourbar.urdf'))
            with open(os.path.join(tmp, 'loop_closures.yaml')) as f:
                config = yaml.safe_load(f)
        self.assertEqual(config['independent'], ['c1_hinge', 'x1_hinge'])
        self.assertEqual(config['dependent'], ['c2_hinge', 'cp_hinge'])

    def test_explicit_dependent_overrides_heuristic(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._four_bar(tmp)
            loop = [c for c in assembly.connections if c.loop][0]
            loop.dependent = ('c1_hinge', 'cp_hinge')
            root = ET.parse(assembly.build(
                output_path=os.path.join(tmp, 'fourbar.urdf'))).getroot()
            with open(os.path.join(tmp, 'loop_closures.yaml')) as f:
                config = yaml.safe_load(f)
        self.assertEqual(config['independent'], ['c2_hinge'])
        self.assertEqual(config['dependent'], ['c1_hinge', 'cp_hinge'])
        mimics = {j.get('name'): j.find('mimic')
                  for j in root.findall('joint')
                  if j.find('mimic') is not None}
        for name, multiplier in (('c1_hinge', 1.0), ('cp_hinge', -1.0)):
            self.assertEqual(mimics[name].get('joint'), 'c2_hinge')
            self.assertAlmostEqual(
                float(mimics[name].get('multiplier')), multiplier)

    def test_explicit_dependent_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._four_bar(tmp)
            with self.assertRaises(ValueError):
                assembly.connect('c1', 'arm', 'c2', 'arm',
                                 dependent=('c1_hinge',))  # without loop
            loop = [c for c in assembly.connections if c.loop][0]
            loop.dependent = ('not_a_ring_joint',)
            with self.assertRaisesRegex(ValueError, 'not movable joints'):
                assembly.build(output_path=os.path.join(tmp, 'x.urdf'))

    def test_chained_parallelograms_propagate_mimic(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            ground = self._write(tmp, 'ground3', _GROUND3_URDF)
            crank = self._write(tmp, 'crank',
                                _BAR_URDF.format(tip_xyz='0 0.1 0',
                                                 tip_rpy='0 0 0'))
            coupler = self._write(tmp, 'coupler',
                                  _BAR_URDF.format(tip_xyz='0.2 0 0',
                                                   tip_rpy='0 0 0'))
            assembly = RobotAssembly('double')
            assembly.add_module_instance('g', ground)
            for cid in ('c1', 'c2', 'c3'):
                assembly.add_module_instance(cid, crank)
            for pid in ('cpA', 'cpB'):
                assembly.add_module_instance(pid, coupler)
            assembly.connect('g', 'g1', 'c1', 'base_link')
            assembly.connect('g', 'g2', 'c2', 'base_link')
            assembly.connect('g', 'g3', 'c3', 'base_link')
            assembly.connect('c1', 'tip', 'cpA', 'base_link')
            assembly.connect('c2', 'tip', 'cpB', 'base_link')
            assembly.connect('cpA', 'tip', 'c2', 'tip', loop=True)
            assembly.connect('cpB', 'tip', 'c3', 'tip', loop=True)
            assembly.set_root('g', 'base_link')
            root = ET.parse(assembly.build(
                output_path=os.path.join(tmp, 'double.urdf'))).getroot()
            with open(os.path.join(tmp, 'loop_closures.yaml')) as f:
                config = yaml.safe_load(f)
        # the whole chain has ONE degree of freedom: everything follows c1
        self.assertEqual(config['independent'], ['c1_hinge'])
        self.assertEqual(config['dependent'],
                         ['c2_hinge', 'c3_hinge', 'cpA_hinge', 'cpB_hinge'])
        mimics = {j.get('name'): j.find('mimic')
                  for j in root.findall('joint')
                  if j.find('mimic') is not None}
        expected = {'c2_hinge': 1.0, 'cpA_hinge': -1.0,
                    'c3_hinge': 1.0, 'cpB_hinge': -1.0}
        self.assertEqual(sorted(mimics), sorted(expected))
        for name, multiplier in expected.items():
            self.assertEqual(mimics[name].get('joint'), 'c1_hinge')
            self.assertAlmostEqual(
                float(mimics[name].get('multiplier')), multiplier)

    def test_five_bar_needs_and_takes_explicit_dependent(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            ground = self._write(tmp, 'ground',
                                 _GROUND_URDF.format(g2_xyz='0.2 0 0'))
            crank = self._write(tmp, 'crank',
                                _BAR_URDF.format(tip_xyz='0 0.1 0',
                                                 tip_rpy='0 0 0'))
            fore_a = self._write(tmp, 'fore_a',
                                 _BAR_URDF.format(tip_xyz='0.1 0 0',
                                                  tip_rpy='0 0 0'))
            fore_b = self._write(tmp, 'fore_b',
                                 _BAR_URDF.format(tip_xyz='-0.1 0 0',
                                                  tip_rpy='0 0 0'))
            assembly = RobotAssembly('fivebar')
            assembly.add_module_instance('g', ground)
            assembly.add_module_instance('c1', crank)
            assembly.add_module_instance('c2', crank)
            assembly.add_module_instance('fa', fore_a)
            assembly.add_module_instance('fb', fore_b)
            assembly.connect('g', 'g1', 'c1', 'base_link')
            assembly.connect('g', 'g2', 'c2', 'base_link')
            assembly.connect('c1', 'tip', 'fa', 'base_link')
            assembly.connect('c2', 'tip', 'fb', 'base_link')
            assembly.connect('fa', 'tip', 'fb', 'tip', loop=True,
                             dependent=('fa_hinge', 'fb_hinge'))
            assembly.set_root('g', 'base_link')
            root = ET.parse(assembly.build(
                output_path=os.path.join(tmp, 'fivebar.urdf'))).getroot()
            with open(os.path.join(tmp, 'loop_closures.yaml')) as f:
                config = yaml.safe_load(f)
        # 2-DOF loop: both cranks stay driven, both forearms are solved
        self.assertEqual(config['independent'], ['c1_hinge', 'c2_hinge'])
        self.assertEqual(config['dependent'], ['fa_hinge', 'fb_hinge'])
        self.assertEqual([j.get('name') for j in root.findall('joint')
                          if j.find('mimic') is not None], [])

    def test_resolve_mimic_chain_composes_linearly(self):
        from lxml import etree

        robot = etree.fromstring(
            '<robot name="r">'
            '<joint name="a" type="revolute">'
            '<parent link="w"/><child link="x"/>'
            '<mimic joint="b" multiplier="2" offset="3"/></joint>'
            '<joint name="b" type="revolute">'
            '<parent link="x"/><child link="y"/>'
            '<mimic joint="c" multiplier="5" offset="7"/></joint>'
            '<joint name="c" type="revolute">'
            '<parent link="y"/><child link="z"/></joint>'
            '</robot>')
        chain = RobotAssembly._resolve_mimic_chain(robot)
        # a = 2*b + 3 and b = 5*c + 7  =>  a = 10*c + 17
        self.assertEqual(chain['a'], ('c', 10.0, 17.0))
        self.assertEqual(chain['b'], ('c', 5.0, 7.0))
        self.assertNotIn('c', chain)

    def test_open_zero_pose_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            # skip the coupler: the two crank tips sit 0.2 m apart at zero
            ground = self._write(tmp, 'ground',
                                 _GROUND_URDF.format(g2_xyz='0.2 0 0'))
            crank = self._write(tmp, 'crank',
                                _BAR_URDF.format(tip_xyz='0 0.1 0',
                                                 tip_rpy='0 0 0'))
            assembly = RobotAssembly('open')
            assembly.add_module_instance('g', ground)
            assembly.add_module_instance('c1', crank)
            assembly.add_module_instance('c2', crank)
            assembly.connect('g', 'g1', 'c1', 'base_link')
            assembly.connect('g', 'g2', 'c2', 'base_link')
            assembly.connect('c1', 'tip', 'c2', 'tip', loop=True)
            assembly.set_root('g', 'base_link')
            with self.assertRaisesRegex(ValueError, 'apart'):
                assembly.build(output_path=os.path.join(tmp, 'x.urdf'))

    def test_loop_round_trips_through_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._four_bar(tmp)
            rebuilt = RobotAssembly.from_dict(assembly.to_dict())
            self.assertEqual(rebuilt.to_dict(), assembly.to_dict())
            loops = [c for c in rebuilt.connections if c.loop]
            self.assertEqual(len(loops), 1)

    def test_build_robot_model_with_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = self._four_bar(tmp)
            robot = assembly.build_robot_model()
        joint_names = {j.name for j in robot.joint_list}
        self.assertIn('c1_hinge', joint_names)
