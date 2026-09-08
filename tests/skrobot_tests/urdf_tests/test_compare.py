"""Tests for skrobot.urdf.compare_kinematics."""

import os
import tempfile
import unittest

from skrobot.urdf import change_urdf_root_link
from skrobot.urdf import compare_kinematics
from skrobot.urdf import merge_fixed_links_file


_VISUAL = ('<visual><origin xyz="0.05 0.01 0" rpy="0.2 0 0"/>'
           '<geometry><box size="0.02 0.02 0.02"/></geometry></visual>')


def _urdf(limit='lower="-0.2" upper="1.4"', axis='0 0 1', visual=_VISUAL):
    return """<?xml version="1.0"?>
<robot name="m">
  <link name="base">{v}</link>
  <link name="arm">{v}</link>
  <link name="tip">{v}</link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="{a}"/>
    <limit {l} effort="1" velocity="1"/>
  </joint>
  <joint name="fix" type="fixed">
    <parent link="arm"/><child link="tip"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
  </joint>
</robot>""".format(v=visual, a=axis, l=limit)


def _write(directory, name, text):
    path = os.path.join(directory, name)
    with open(path, 'w') as f:
        f.write(text)
    return path


class TestComparePreservingTransforms(unittest.TestCase):
    """A transform that is supposed to preserve the mechanism reads as
    equivalent, even though it rewrites the document heavily."""

    def test_merge_fixed_links_is_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(tmp, 'a.urdf', _urdf())
            dst = os.path.join(tmp, 'b.urdf')
            merge_fixed_links_file(src, dst)
            result = compare_kinematics(dst, src)
            self.assertTrue(result.is_equivalent, str(result))
            self.assertEqual(result.only_in_reference, ['tip'])

    def test_change_root_link_is_equivalent(self):
        """Re-rooting moves every link frame and compensates in the geometry
        origins, so comparing frames would call it broken; comparing geometry
        does not."""
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(tmp, 'a.urdf', _urdf())
            dst = os.path.join(tmp, 'b.urdf')
            change_urdf_root_link(src, 'tip', dst)
            result = compare_kinematics(dst, src)
            self.assertTrue(result.is_equivalent, str(result))
            self.assertLess(result.max_position_error, 1e-12)

    def test_a_document_equals_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(tmp, 'a.urdf', _urdf())
            self.assertTrue(compare_kinematics(src, src).is_equivalent)


class TestCompareCatchesRealChanges(unittest.TestCase):

    def test_a_negated_axis_alone_is_not_equivalent(self):
        """Negating <axis> without moving the limits reverses the joint, which
        a canonical text diff would report but could not distinguish from the
        harmless sign choices it also reports."""
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(tmp, 'a.urdf', _urdf())
            dst = _write(tmp, 'b.urdf', _urdf(axis='0 0 -1'))
            result = compare_kinematics(dst, src)
            self.assertFalse(result.is_equivalent)
            self.assertTrue(any('moves differently' in d
                                for d in result.differences), result)

    def test_a_moved_joint_origin_is_not_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(tmp, 'a.urdf', _urdf())
            dst = _write(tmp, 'b.urdf',
                         _urdf().replace('xyz="0 0 0.1"', 'xyz="0 0 0.11"'))
            result = compare_kinematics(dst, src)
            self.assertFalse(result.is_equivalent)
            self.assertAlmostEqual(result.max_position_error, 0.01, places=9)

    def test_differing_limits_are_reported_but_do_not_skew_the_poses(self):
        """The requested angle is clamped into BOTH documents' limits, so a
        differing limit shows up as itself rather than as everything below the
        joint appearing to move."""
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(tmp, 'a.urdf', _urdf())
            dst = _write(tmp, 'b.urdf', _urdf(limit='lower="-0.2" upper="0.9"'))
            result = compare_kinematics(dst, src)
            self.assertFalse(result.is_equivalent)
            self.assertEqual(result.max_position_error, 0.0)
            self.assertTrue(any('max_angle differs' in d
                                for d in result.differences), result)
            self.assertTrue(
                compare_kinematics(dst, src, compare_limits=False)
                .is_equivalent)


class TestCompareReporting(unittest.TestCase):

    def test_a_link_without_geometry_is_not_silently_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = _urdf().replace('<link name="tip">{}</link>'.format(_VISUAL),
                                   '<link name="tip"/>')
            src = _write(tmp, 'a.urdf', text)
            result = compare_kinematics(src, src)
            self.assertIn('tip', result.not_comparable)
            self.assertNotIn('tip/visual#0', result.compared)

    def test_no_shared_geometry_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(tmp, 'a.urdf', _urdf())
            other = _write(tmp, 'b.urdf', _urdf().replace('name="m"', 'name="o"')
                           .replace('"base"', '"b2"').replace('"arm"', '"a2"')
                           .replace('"tip"', '"t2"'))
            with self.assertRaises(ValueError):
                compare_kinematics(other, src)

    def test_per_joint_angles_may_be_given_as_a_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(tmp, 'a.urdf', _urdf())
            dst = _write(tmp, 'b.urdf', _urdf(axis='0 0 -1'))
            self.assertTrue(
                compare_kinematics(dst, src, joint_angles=({'j1': 0.0},),
                                   compare_limits=False).is_equivalent)
            self.assertFalse(
                compare_kinematics(dst, src, joint_angles=({'j1': 0.4},),
                                   compare_limits=False).is_equivalent)

    def test_summary_names_what_moved(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(tmp, 'a.urdf', _urdf())
            dst = _write(tmp, 'b.urdf', _urdf(axis='0 0 -1'))
            text = str(compare_kinematics(dst, src))
            self.assertIn('Kinematic Comparison', text)
            self.assertIn('Equivalent: NO', text)
            self.assertIn('arm/visual#0', text)


if __name__ == '__main__':
    unittest.main()
