"""Tests for the `urdf-tree` CLI app (skrobot.apps.urdf_tree)."""

import json
import subprocess
import sys
import unittest

import pytest

from skrobot.data import kuka_urdfpath


BAD_URDF = (
    '<robot name="bad">'
    '<link name="a"/><link name="b"/><link name="c"/>'
    '<joint name="j" type="fixed">'
    '<parent link="a"/><child link="b"/></joint>'
    '</robot>'
)


def _run(args, stdin=None):
    return subprocess.run(
        [sys.executable, '-m', 'skrobot.apps.urdf_tree'] + args,
        input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding='utf-8')


@pytest.mark.skipif(
    sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
    reason="Skip in Python 2 and Python 3.6")
class TestUrdfTreeApp(unittest.TestCase):

    def test_valid_urdf_default(self):
        proc = _run([kuka_urdfpath(), '--no-color'])
        self.assertEqual(proc.returncode, 0)
        self.assertIn('URDF Validation Summary:', proc.stdout)
        self.assertIn('All validation checks passed!', proc.stdout)
        self.assertIn('Kinematic Tree', proc.stdout)

    def test_full_tree(self):
        proc = _run([kuka_urdfpath(), '--full', '--tree-only', '--no-color'])
        self.assertEqual(proc.returncode, 0)
        self.assertIn('URDF Link Tree Structure:', proc.stdout)

    def test_invalid_urdf_from_stdin_exit_code(self):
        proc = _run(['-', '--validate-only', '--no-color'], stdin=BAD_URDF)
        self.assertEqual(proc.returncode, 1)
        self.assertIn('Validation Errors:', proc.stdout)

    def test_json_output(self):
        proc = _run([kuka_urdfpath(), '--json'])
        self.assertEqual(proc.returncode, 0)
        data = json.loads(proc.stdout)
        self.assertTrue(data['is_valid'])
        self.assertEqual(data['summary']['joints_count'], 14)

    def test_world_mode(self):
        proc = _run([kuka_urdfpath(), '--tree-only', '--world', '--no-color'])
        self.assertEqual(proc.returncode, 0)
        self.assertIn('world axes', proc.stdout)
        self.assertIn('@[', proc.stdout)

    def test_missing_file(self):
        proc = _run(['/no/such/file.urdf'])
        self.assertEqual(proc.returncode, 2)


if __name__ == '__main__':
    unittest.main()
