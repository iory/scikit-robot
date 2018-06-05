import unittest

import robot


class TestRobotModel(unittest.TestCase):

    def test_init(self):
        fetch = robot.robots.Fetch()
        fetch.angle_vector()
