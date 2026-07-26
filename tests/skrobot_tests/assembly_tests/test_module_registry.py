import os
import tempfile
import unittest

from skrobot.assembly import ModuleRegistry


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


class TestModuleRegistry(unittest.TestCase):

    def test_scan_registers_every_urdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_module(tmp, 'a')
            _write_module(tmp, 'b')

            registry = ModuleRegistry(tmp)

            self.assertEqual(registry.scan(), 2)
            self.assertEqual(sorted(registry.list_modules()), ['a', 'b'])

    def test_register_urdf_adds_a_single_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_module(tmp, 'later')
            registry = ModuleRegistry(tmp)

            module_id = registry.register_urdf(path)

            self.assertEqual(module_id, 'later')
            self.assertIn('later', registry)
            self.assertIsNotNone(registry.get_module('later'))
            self.assertEqual(len(registry), 1)

    def test_register_urdf_replaces_a_module_with_the_same_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_module(tmp, 'dup')
            registry = ModuleRegistry(tmp)

            registry.register_urdf(path)
            first = registry.get_module('dup')
            registry.register_urdf(path)

            self.assertEqual(len(registry), 1)
            self.assertIsNot(registry.get_module('dup'), first)

    def test_remove_module_reports_whether_it_was_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_module(tmp, 'gone')
            registry = ModuleRegistry(tmp)
            registry.register_urdf(path)

            self.assertTrue(registry.remove_module('gone'))
            self.assertNotIn('gone', registry)
            self.assertEqual(len(registry), 0)
            self.assertFalse(registry.remove_module('gone'))
