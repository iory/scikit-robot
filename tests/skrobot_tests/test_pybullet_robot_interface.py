import unittest

import skrobot
from skrobot.pybullet_robot_interface import PybulletRobotInterface


class TestPybulletRobotInterface(unittest.TestCase):

    def test_init(self):
        fetch = skrobot.models.Fetch()
        PybulletRobotInterface(fetch, connect=2)
