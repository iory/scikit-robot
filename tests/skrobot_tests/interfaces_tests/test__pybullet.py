import unittest

import skrobot
from skrobot.interfaces._pybullet import PybulletRobotInterface


class TestPybulletRobotInterface(unittest.TestCase):

    def test_init(self):
        fetch = skrobot.models.Fetch()
        PybulletRobotInterface(fetch, connect=2)
