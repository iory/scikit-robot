import unittest

import robot
from robot.pybullet_robot_interface import PybulletRobotInterface


class TestPybulletRobotInterface(unittest.TestCase):

    def test_init(self):
        fetch = robot.robots.Fetch()
        pri = PybulletRobotInterface(fetch, connect=2)
