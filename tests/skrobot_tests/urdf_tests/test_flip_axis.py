import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from skrobot.urdf import change_urdf_root_link
from skrobot.urdf import flip_joint_axis
from skrobot.urdf import flip_joint_axis_file


_URDF = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="mid"/>
  <link name="tip"/>
  <joint name="drive" type="revolute">
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="base"/><child link="mid"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="2.0" effort="10" velocity="1"/>
  </joint>
  <joint name="follow" type="revolute">
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <parent link="mid"/><child link="tip"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="10" velocity="1"/>
    <mimic joint="drive" multiplier="2.0" offset="0.25"/>
  </joint>
</robot>"""


def _joint(root, name):
    return next(j for j in root.findall('joint') if j.get('name') == name)


class TestFlipJointAxis(unittest.TestCase):

    def test_axis_and_limits_move_together(self):
        root = ET.fromstring(_URDF)
        self.assertTrue(flip_joint_axis(root, 'drive'))
        drive = _joint(root, 'drive')
        self.assertEqual(drive.find('axis').get('xyz'), '0 0 -1')
        # the reachable range is preserved, expressed in the new sense
        self.assertEqual(drive.find('limit').get('lower'), '-2')
        self.assertEqual(drive.find('limit').get('upper'), '0.5')

    def test_origin_is_untouched(self):
        root = ET.fromstring(_URDF)
        before = _joint(root, 'drive').find('origin').attrib.copy()
        flip_joint_axis(root, 'drive')
        self.assertEqual(_joint(root, 'drive').find('origin').attrib, before)

    def test_followers_of_the_flipped_joint_keep_phase(self):
        """A joint mimicking the flipped one needs -multiplier, same offset."""
        root = ET.fromstring(_URDF)
        flip_joint_axis(root, 'drive')
        mimic = _joint(root, 'follow').find('mimic')
        self.assertEqual(mimic.get('multiplier'), '-2')
        self.assertEqual(mimic.get('offset'), '0.25')
        # the follower's own axis and limits are its own business
        self.assertEqual(_joint(root, 'follow').find('axis').get('xyz'),
                         '0 1 0')

    def test_flipping_a_follower_negates_both_its_coefficients(self):
        root = ET.fromstring(_URDF)
        flip_joint_axis(root, 'follow')
        mimic = _joint(root, 'follow').find('mimic')
        self.assertEqual(mimic.get('multiplier'), '-2')
        self.assertEqual(mimic.get('offset'), '-0.25')
        # the driver it points at is unaffected
        self.assertEqual(_joint(root, 'drive').find('axis').get('xyz'),
                         '0 0 1')

    def test_flip_is_its_own_inverse(self):
        root = ET.fromstring(_URDF)
        before = ET.tostring(root)
        flip_joint_axis(root, 'drive')
        self.assertNotEqual(ET.tostring(root), before)
        flip_joint_axis(root, 'drive')
        after = ET.fromstring(ET.tostring(root))
        original = ET.fromstring(_URDF)
        for name in ('drive', 'follow'):
            for tag in ('axis', 'limit', 'mimic'):
                got, want = _joint(after, name).find(tag), \
                    _joint(original, name).find(tag)
                if want is None:
                    continue
                for key, value in want.attrib.items():
                    where = '{}/{}/{}'.format(name, tag, key)
                    try:
                        numbers = [float(v) for v in value.split()]
                    except ValueError:
                        self.assertEqual(got.get(key), value, msg=where)
                        continue
                    self.assertEqual(len(got.get(key).split()), len(numbers),
                                     msg=where)
                    for a, b in zip(got.get(key).split(), numbers):
                        self.assertAlmostEqual(float(a), b, msg=where)

    def test_one_sided_limit_changes_side_not_just_sign(self):
        """``q <= 0.4`` is ``q' >= -0.4``: the bound moves to the other
        attribute.  Keeping it as an upper bound would silently relocate the
        reachable range instead of mirroring it."""
        root = ET.fromstring("""<robot name="r">
          <joint name="j" type="prismatic"><axis xyz="1 0 0"/>
            <limit upper="0.4" effort="1" velocity="1"/></joint>
        </robot>""")
        self.assertTrue(flip_joint_axis(root, 'j'))
        limit = root.find('joint').find('limit')
        self.assertEqual(limit.get('lower'), '-0.4')
        self.assertIsNone(limit.get('upper'))
        # and it is still self-inverse
        flip_joint_axis(root, 'j')
        limit = root.find('joint').find('limit')
        self.assertEqual(limit.get('upper'), '0.4')
        self.assertIsNone(limit.get('lower'))

    def test_safety_controller_soft_limits_move_with_the_limit(self):
        root = ET.fromstring("""<robot name="r">
          <joint name="j" type="revolute"><axis xyz="0 0 1"/>
            <limit lower="-0.5" upper="2.0" effort="1" velocity="1"/>
            <safety_controller k_velocity="1" soft_lower_limit="-0.4"
              soft_upper_limit="1.9"/></joint>
        </robot>""")
        self.assertTrue(flip_joint_axis(root, 'j'))
        safety = root.find('joint').find('safety_controller')
        self.assertEqual(safety.get('soft_lower_limit'), '-1.9')
        self.assertEqual(safety.get('soft_upper_limit'), '0.4')
        # unrelated attributes are untouched
        self.assertEqual(safety.get('k_velocity'), '1')

    def test_implicit_mimic_multiplier_is_written_out(self):
        """``<mimic>`` without a multiplier means 1.0; after the driver flips
        the follower needs -1, which has to become explicit."""
        root = ET.fromstring("""<robot name="r">
          <joint name="drive" type="revolute"><axis xyz="0 0 1"/></joint>
          <joint name="follow" type="revolute"><axis xyz="0 0 1"/>
            <mimic joint="drive"/></joint>
        </robot>""")
        self.assertTrue(flip_joint_axis(root, 'drive'))
        mimic = _joint(root, 'follow').find('mimic')
        self.assertEqual(mimic.get('multiplier'), '-1')

    def test_flipping_a_follower_with_implicit_multiplier(self):
        root = ET.fromstring("""<robot name="r">
          <joint name="drive" type="revolute"><axis xyz="0 0 1"/></joint>
          <joint name="follow" type="revolute"><axis xyz="0 0 1"/>
            <mimic joint="drive" offset="0.25"/></joint>
        </robot>""")
        self.assertTrue(flip_joint_axis(root, 'follow'))
        mimic = _joint(root, 'follow').find('mimic')
        self.assertEqual(mimic.get('multiplier'), '-1')
        self.assertEqual(mimic.get('offset'), '-0.25')

    def test_returns_false_without_a_usable_axis(self):
        root = ET.fromstring("""<robot name="r">
          <joint name="fixed_j" type="fixed"/>
          <joint name="zero_j" type="revolute"><axis xyz="0 0 0"/></joint>
        </robot>""")
        self.assertFalse(flip_joint_axis(root, 'fixed_j'))
        self.assertFalse(flip_joint_axis(root, 'zero_j'))
        self.assertFalse(flip_joint_axis(root, 'no_such_joint'))

    def test_continuous_joint_without_limit(self):
        root = ET.fromstring("""<robot name="r">
          <joint name="spin" type="continuous"><axis xyz="0 0 1"/></joint>
        </robot>""")
        self.assertTrue(flip_joint_axis(root, 'spin'))
        self.assertEqual(root.find('joint').find('axis').get('xyz'), '0 0 -1')


class TestFlipJointAxisFile(unittest.TestCase):

    def test_in_place_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'r.urdf')
            with open(path, 'w') as f:
                f.write(_URDF)
            self.assertTrue(flip_joint_axis_file(path, 'drive'))
            root = ET.parse(path).getroot()
            self.assertEqual(_joint(root, 'drive').find('axis').get('xyz'),
                             '0 0 -1')

    def test_missing_joint_leaves_the_file_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'r.urdf')
            with open(path, 'w') as f:
                f.write(_URDF)
            self.assertFalse(flip_joint_axis_file(path, 'nope'))
            with open(path) as f:
                self.assertEqual(f.read(), _URDF)


class TestRootLinkChangeKeepsLimits(unittest.TestCase):
    """Re-rooting negates the axis of every joint along the path, and must
    leave the limits alone.

    Two reversals happen there, not one: swapping <parent>/<child> already
    reverses the joint's sense, so negating <axis> puts the sense back.  The
    same commanded value therefore still produces the same pose, and remapping
    the limits as well would send the joint through its range mirrored about
    zero -- shrinking the reach on one side and inventing it on the other.
    A standalone flip_joint_axis is the case that DOES have to remap them; the
    tests above cover it."""

    def test_asymmetric_limits_survive_the_reversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'in.urdf')
            dst = os.path.join(tmp, 'out.urdf')
            with open(src, 'w') as f:
                f.write(_URDF)
            change_urdf_root_link(src, 'mid', dst)

            drive = _joint(ET.parse(dst).getroot(), 'drive')
            axis = [float(v) for v in drive.find('axis').get('xyz').split()]
            self.assertEqual(axis, [0.0, 0.0, -1.0])
            limit = drive.find('limit')
            self.assertAlmostEqual(float(limit.get('lower')), -0.5)
            self.assertAlmostEqual(float(limit.get('upper')), 2.0)


if __name__ == '__main__':
    unittest.main()


class TestRootLinkChangeKeepsSafetyAndMimic(unittest.TestCase):
    """The same cancellation applies to everything else expressed in the
    joint's own value: the soft bounds on <safety_controller>, the reversed
    joint's own <mimic>, and the <mimic> of a follower that is NOT on the
    re-rooting path but tracks a joint that is."""

    _URDF_MIMIC = """<?xml version="1.0"?>
<robot name="m">
  <link name="base"/><link name="mid"/><link name="tip"/><link name="finger"/>
  <joint name="drive" type="revolute">
    <parent link="base"/><child link="mid"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="2.0" effort="10" velocity="1"/>
    <safety_controller soft_lower_limit="-0.4" soft_upper_limit="1.9"
                       k_position="100" k_velocity="10"/>
  </joint>
  <joint name="fix" type="fixed">
    <parent link="mid"/><child link="tip"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="follower" type="revolute">
    <parent link="mid"/><child link="finger"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="10" velocity="1"/>
    <mimic joint="drive" multiplier="2.5" offset="0.1"/>
  </joint>
</robot>"""

    def test_soft_bounds_and_mimic_survive_the_reversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'in.urdf')
            dst = os.path.join(tmp, 'out.urdf')
            with open(src, 'w') as f:
                f.write(self._URDF_MIMIC)
            change_urdf_root_link(src, 'tip', dst)
            root = ET.parse(dst).getroot()

            drive = _joint(root, 'drive')
            self.assertEqual(
                [float(v) for v in drive.find('axis').get('xyz').split()],
                [0.0, 0.0, -1.0])
            safety = drive.find('safety_controller')
            self.assertAlmostEqual(
                float(safety.get('soft_lower_limit')), -0.4)
            self.assertAlmostEqual(
                float(safety.get('soft_upper_limit')), 1.9)

            # 'follower' hangs off the path rather than lying on it, so it is
            # never reversed itself; its coupling to 'drive' is unchanged
            # because 'drive' still means what it meant.
            mimic = _joint(root, 'follower').find('mimic')
            self.assertEqual(mimic.get('joint'), 'drive')
            self.assertAlmostEqual(float(mimic.get('multiplier')), 2.5)
            self.assertAlmostEqual(float(mimic.get('offset')), 0.1)
