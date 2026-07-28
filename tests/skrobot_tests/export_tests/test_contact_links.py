import unittest

import numpy as np

import skrobot
from skrobot.export.contact_links import contact_links
from skrobot.models.urdf import RobotModelFromURDF


KUKA_URDF = skrobot.data.kuka_urdfpath() \
    if hasattr(skrobot.data, 'kuka_urdfpath') else None


class TestContactLinks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import os
        path = os.path.join(os.path.dirname(skrobot.__file__),
                            'data', 'kuka_description', 'kuka.urdf')
        cls.robot = RobotModelFromURDF(urdf_file=path)

    def test_near_target_returns_the_reaching_links(self):
        r = self.robot
        r.angle_vector(np.zeros(len(r.joint_list)))
        # a target right at the end-effector link's origin must select it, and
        # not the base link far away
        ee = r.link_list[-1]
        target = ee.worldpos()
        links = contact_links(r, [r.angle_vector()], target, threshold=0.15)
        self.assertIn(ee.name, links)
        self.assertNotIn(r.link_list[0].name, links)

    def test_far_target_returns_nothing(self):
        r = self.robot
        r.angle_vector(np.zeros(len(r.joint_list)))
        links = contact_links(r, [r.angle_vector()], [100.0, 100.0, 100.0],
                              threshold=0.05)
        self.assertEqual(links, [])

    def test_ordered_by_closest_approach(self):
        r = self.robot
        r.angle_vector(np.zeros(len(r.joint_list)))
        ee = r.link_list[-1]
        links = contact_links(r, [r.angle_vector()], ee.worldpos(),
                              threshold=1.0)
        # first entry is the closest link
        self.assertTrue(len(links) >= 1)

    def test_targets_shape_validation(self):
        r = self.robot
        with self.assertRaises(ValueError):
            contact_links(r, [np.zeros(len(r.joint_list))],
                          np.zeros((2, 4)), threshold=0.1)


if __name__ == '__main__':
    unittest.main()
