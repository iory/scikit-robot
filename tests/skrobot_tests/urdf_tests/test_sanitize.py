import unittest
import xml.etree.ElementTree as ET

from skrobot.urdf import sanitize_name
from skrobot.urdf import sanitize_urdf_names


class TestSanitizeName(unittest.TestCase):

    def test_punctuation_collapses_to_underscore(self):
        self.assertEqual(sanitize_name('base link (rev.2)'), 'base_link_rev_2')
        self.assertEqual(sanitize_name('arm-upper/left'), 'arm_upper_left')

    def test_leading_digit_and_empty(self):
        self.assertEqual(sanitize_name('42_arm'), 'c_42_arm')
        self.assertEqual(sanitize_name('---'), 'c_')

    def test_idempotent_on_valid_names(self):
        for name in ('base_link', 'joint_1', 'c_42'):
            self.assertEqual(sanitize_name(name), name)


class TestSanitizeUrdfNames(unittest.TestCase):

    def test_names_and_cross_references(self):
        root = ET.fromstring("""
<robot name="r">
  <link name="base link"/>
  <link name="arm-1"/>
  <joint name="shoulder joint" type="revolute">
    <parent link="base link"/>
    <child link="arm-1"/>
  </joint>
  <joint name="follower" type="revolute">
    <parent link="base link"/>
    <child link="arm-1"/>
    <mimic joint="shoulder joint" multiplier="2"/>
  </joint>
  <transmission name="tr1">
    <joint name="shoulder joint"><hardwareInterface>EffortJointInterface</hardwareInterface></joint>
  </transmission>
</robot>
""")
        sanitize_urdf_names(root)
        link_names = [link.get('name') for link in root.findall('link')]
        self.assertEqual(link_names, ['base_link', 'arm_1'])
        joint = root.find('joint')
        self.assertEqual(joint.get('name'), 'shoulder_joint')
        self.assertEqual(joint.find('parent').get('link'), 'base_link')
        self.assertEqual(joint.find('child').get('link'), 'arm_1')
        mimic = root.findall('joint')[1].find('mimic')
        self.assertEqual(mimic.get('joint'), 'shoulder_joint')
        transmission_joint = root.find('transmission').find('joint')
        self.assertEqual(transmission_joint.get('name'), 'shoulder_joint')

    def test_colliding_dirty_names_stay_unique(self):
        root = ET.fromstring("""
<robot name="r">
  <link name="arm 1"/>
  <link name="arm-1"/>
  <link name="arm_1"/>
</robot>
""")
        sanitize_urdf_names(root)
        names = sorted(link.get('name') for link in root.findall('link'))
        self.assertEqual(len(set(names)), 3)
        self.assertIn('arm_1', names)

    def test_clean_document_is_untouched(self):
        xml = ('<robot name="r"><link name="base_link"/>'
               '<joint name="j1" type="fixed">'
               '<parent link="base_link"/><child link="base_link"/>'
               '</joint></robot>')
        root = ET.fromstring(xml)
        sanitize_urdf_names(root)
        self.assertEqual(root.find('link').get('name'), 'base_link')
        self.assertEqual(root.find('joint').get('name'), 'j1')


if __name__ == '__main__':
    unittest.main()
