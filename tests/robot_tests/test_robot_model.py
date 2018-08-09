import unittest

import robot


class TestRobotModel(unittest.TestCase):

    def test_init(self):
        fetch = robot.robots.Fetch()
        fetch.angle_vector()

    def test_find_link_route(self):
        fetch = robot.robots.Fetch()
        ret = fetch.find_link_route(fetch.torso_lift_link)
        self.assertEqual(ret,
                         [fetch.torso_lift_link])

        ret = fetch.find_link_route(fetch.wrist_roll_link)
        self.assertEqual(ret,
                         [fetch.torso_lift_link,
                          fetch.shoulder_pan_link,
                          fetch.shoulder_lift_link,
                          fetch.upperarm_roll_link,
                          fetch.elbow_flex_link,
                          fetch.forearm_roll_link,
                          fetch.wrist_flex_link,
                          fetch.wrist_roll_link])
