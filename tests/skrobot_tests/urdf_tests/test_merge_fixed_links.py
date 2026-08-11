import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from skrobot.urdf import merge_fixed_links
from skrobot.urdf import merge_fixed_links_file


_URDF = """<?xml version="1.0"?>
<robot name="t">
  <link name="base">
    <visual><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="1 1 1"/></geometry></visual>
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="2"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <joint name="fix_c" type="fixed">
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <parent link="base"/><child link="c"/>
  </joint>
  <link name="c">
    <visual><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.5 0.5 0.5"/></geometry></visual>
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="3"/>
      <inertia ixx="2" ixy="0" ixz="0" iyy="2" iyz="0" izz="2"/></inertial>
  </link>
  <joint name="mov" type="revolute">
    <origin xyz="0 1 0" rpy="0 0 0"/>
    <parent link="c"/><child link="g"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <link name="g"><visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual></link>
  <joint name="fix_frame" type="fixed">
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <parent link="base"/><child link="frame"/>
  </joint>
  <link name="frame"/>
</robot>"""


def _names(root, tag):
    return [e.get('name') for e in root.findall(tag)]


def _floats(text):
    return [float(v) for v in text.split()]


class TestMergeFixedLinks(unittest.TestCase):

    def test_geometry_moves_with_a_composed_origin(self):
        root = ET.fromstring(_URDF)
        self.assertEqual(merge_fixed_links(root), 1)
        self.assertEqual(_names(root, 'link'), ['base', 'g', 'frame'])
        base = root.find('link')
        visuals = base.findall('visual')
        self.assertEqual(len(visuals), 2)
        # the merged child's visual carries the fixed joint's translation
        self.assertEqual(_floats(visuals[1].find('origin').get('xyz')),
                         [1.0, 0.0, 0.0])

    def test_movable_joint_is_reparented_and_its_frame_preserved(self):
        root = ET.fromstring(_URDF)
        merge_fixed_links(root)
        mov = next(j for j in root.findall('joint') if j.get('name') == 'mov')
        self.assertEqual(mov.find('parent').get('link'), 'base')
        # 'mov' hung 1 m along +y off c, which sat 1 m along +x off base
        self.assertEqual(_floats(mov.find('origin').get('xyz')),
                         [1.0, 1.0, 0.0])

    def test_inertials_combine_about_the_merged_centre_of_mass(self):
        root = ET.fromstring(_URDF)
        merge_fixed_links(root)
        inertial = root.find('link').find('inertial')
        self.assertAlmostEqual(
            float(inertial.find('mass').get('value')), 5.0)
        # 2 kg at the origin and 3 kg at x=1 balance at x=0.6
        self.assertAlmostEqual(
            _floats(inertial.find('origin').get('xyz'))[0], 0.6)
        tensor = inertial.find('inertia')
        # parallel axis about x is unchanged; y and z each gain m*d^2
        self.assertAlmostEqual(float(tensor.get('ixx')), 3.0)
        self.assertAlmostEqual(float(tensor.get('iyy')),
                               1 + 2 * 0.36 + 2 + 3 * 0.16)

    def test_geometryless_frames_are_kept_by_default(self):
        root = ET.fromstring(_URDF)
        merge_fixed_links(root)
        self.assertIn('frame', _names(root, 'link'))
        self.assertIn('fix_frame', _names(root, 'joint'))

    def test_keep_frames_false_collapses_them(self):
        root = ET.fromstring(_URDF)
        merged = merge_fixed_links(root, keep_frames=False)
        self.assertEqual(merged, 2)
        self.assertNotIn('frame', _names(root, 'link'))

    def test_a_chain_of_fixed_joints_collapses_to_a_fixpoint(self):
        root = ET.fromstring("""<robot name="chain">
          <link name="a"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
          <link name="b"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
          <link name="c"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
          <joint name="j1" type="fixed"><origin xyz="1 0 0"/>
            <parent link="a"/><child link="b"/></joint>
          <joint name="j2" type="fixed"><origin xyz="0 2 0"/>
            <parent link="b"/><child link="c"/></joint>
        </robot>""")
        self.assertEqual(merge_fixed_links(root), 2)
        self.assertEqual(_names(root, 'link'), ['a'])
        # c's geometry ends up at the composition of both joint origins
        origins = [_floats(v.find('origin').get('xyz'))
                   for v in root.find('link').findall('visual')[1:]]
        self.assertIn([1.0, 0.0, 0.0], origins)
        self.assertIn([1.0, 2.0, 0.0], origins)

    def test_rotation_is_applied_to_the_child_tensor(self):
        """A yaw of 90 degrees swaps the child's ixx and iyy before they are
        combined, so a dropped or transposed rotation is visible."""
        root = ET.fromstring("""<robot name="rot">
          <link name="base"><visual><geometry><box size="1 1 1"/></geometry></visual>
            <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="2"/>
              <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
          <link name="pcb"><visual><geometry><box size="1 1 1"/></geometry></visual>
            <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="3"/>
              <inertia ixx="2" ixy="0" ixz="0" iyy="5" iyz="0" izz="7"/></inertial></link>
          <joint name="j" type="fixed"><origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
            <parent link="base"/><child link="pcb"/></joint>
        </robot>""")
        merge_fixed_links(root)
        tensor = root.find('link').find('inertial').find('inertia')
        # Rz(90) maps diag(2,5,7) to diag(5,2,7); then parallel axis about x=0.6
        self.assertAlmostEqual(float(tensor.get('ixx')), 6.0, places=6)
        self.assertAlmostEqual(float(tensor.get('iyy')), 4.2, places=6)
        self.assertAlmostEqual(float(tensor.get('izz')), 9.2, places=6)

    def test_force_merge_folds_a_massonly_link(self):
        root = ET.fromstring("""<robot name="m">
          <link name="base"><visual><geometry><box size="1 1 1"/></geometry></visual>
            <inertial><origin xyz="0 0 0"/><mass value="1"/>
              <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
          <link name="weight">
            <inertial><origin xyz="0 0 0"/><mass value="4"/>
              <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
          <joint name="j" type="fixed"><origin xyz="0 0 0"/>
            <parent link="base"/><child link="weight"/></joint>
        </robot>""")
        # without force_merge it is a frame and stays
        untouched = ET.fromstring(ET.tostring(root))
        self.assertEqual(merge_fixed_links(untouched), 0)
        self.assertEqual(merge_fixed_links(root, force_merge={'weight'}), 1)
        self.assertAlmostEqual(
            float(root.find('link').find('inertial').find('mass').get('value')),
            5.0)

    def test_only_restricts_folding_to_the_named_child(self):
        root = ET.fromstring(_URDF)
        self.assertEqual(merge_fixed_links(root, only={'nothing'}), 0)
        self.assertEqual(_names(root, 'link'), ['base', 'c', 'g', 'frame'])

    def test_a_link_without_inertial_is_handled(self):
        root = ET.fromstring("""<robot name="n">
          <link name="a"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
          <link name="b"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
          <joint name="j" type="fixed"><origin xyz="1 0 0"/>
            <parent link="a"/><child link="b"/></joint>
        </robot>""")
        self.assertEqual(merge_fixed_links(root), 1)
        self.assertIsNone(root.find('link').find('inertial'))

    def test_world_poses_are_unchanged(self):
        """The point of the transform: every link that survives keeps the pose
        it had, so nothing downstream moves."""
        def poses(robot):
            placement = {}
            edges = [(j.find('parent').get('link'), j.find('child').get('link'),
                      parse(j)) for j in robot.findall('joint')]
            names = [ln.get('name') for ln in robot.findall('link')]
            for name in names:
                matrix, current = np.eye(4), name
                while True:
                    step = next((e for e in edges if e[1] == current), None)
                    if step is None:
                        break
                    matrix, current = step[2].dot(matrix), step[0]
                placement[name] = matrix
            return placement

        from skrobot.utils.urdf import parse_origin as parse
        root = ET.fromstring(_URDF)
        before = poses(ET.fromstring(_URDF))
        merge_fixed_links(root)
        after = poses(root)
        for name, matrix in after.items():
            np.testing.assert_allclose(matrix, before[name], atol=1e-12,
                                       err_msg=name)


class TestMalformedTopology(unittest.TestCase):
    """A merge that would leave the tree inconsistent must not happen."""

    def test_a_self_loop_does_not_delete_its_link(self):
        root = ET.fromstring("""<robot name="r">
          <link name="a"><visual><geometry><box size="1 1 1"/></geometry>
            </visual></link>
          <joint name="loop" type="fixed"><origin xyz="1 1 1"/>
            <parent link="a"/><child link="a"/></joint>
        </robot>""")
        self.assertEqual(merge_fixed_links(root), 0)
        self.assertEqual(_names(root, 'link'), ['a'])

    def test_a_link_with_two_parents_is_left_alone(self):
        """Folding it into one parent would leave the other joint pointing
        at a link that no longer exists."""
        root = ET.fromstring("""<robot name="r">
          <link name="a"><visual><geometry><box size="1 1 1"/></geometry>
            </visual></link>
          <link name="b"><visual><geometry><box size="1 1 1"/></geometry>
            </visual></link>
          <link name="c"><visual><geometry><box size="1 1 1"/></geometry>
            </visual></link>
          <joint name="j1" type="fixed"><origin xyz="1 0 0"/>
            <parent link="a"/><child link="c"/></joint>
          <joint name="j2" type="fixed"><origin xyz="0 9 0"/>
            <parent link="b"/><child link="c"/></joint>
        </robot>""")
        self.assertEqual(merge_fixed_links(root), 0)
        self.assertEqual(_names(root, 'link'), ['a', 'b', 'c'])
        children = {j.find('child').get('link')
                    for j in root.findall('joint')}
        self.assertTrue(children <= set(_names(root, 'link')))

    def test_only_still_folds_with_an_empty_force_merge(self):
        """Naming a child in ``only`` is the request to fold it; passing an
        empty ``force_merge`` (a common default) must not cancel that."""
        urdf = """<robot name="m">
          <link name="base"><visual><geometry><box size="1 1 1"/></geometry>
            </visual></link>
          <link name="weight"><inertial><origin xyz="0 0 0"/>
            <mass value="4"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
            </inertial></link>
          <joint name="j" type="fixed"><origin xyz="0 0 0"/>
            <parent link="base"/><child link="weight"/></joint>
        </robot>"""
        root = ET.fromstring(urdf)
        self.assertEqual(merge_fixed_links(root, only={'weight'},
                                           force_merge=[]), 1)
        self.assertEqual(_names(root, 'link'), ['base'])


class TestMergeFixedLinksFile(unittest.TestCase):

    def test_round_trip_keeps_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'r.urdf')
            with open(path, 'w') as f:
                f.write(_URDF.replace('<link name="base">',
                                      '<!-- provenance --><link name="base">'))
            self.assertEqual(merge_fixed_links_file(path), 1)
            with open(path) as f:
                text = f.read()
            self.assertIn('<!-- provenance -->', text)
            self.assertTrue(text.startswith('<?xml'))

    def test_writes_to_a_separate_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'in.urdf')
            dst = os.path.join(tmp, 'out.urdf')
            with open(src, 'w') as f:
                f.write(_URDF)
            merge_fixed_links_file(src, dst)
            with open(src) as f:
                self.assertIn('name="c"', f.read())
            self.assertEqual(
                _names(ET.parse(dst).getroot(), 'link'),
                ['base', 'g', 'frame'])


if __name__ == '__main__':
    unittest.main()
