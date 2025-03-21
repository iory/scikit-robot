import unittest

import pytest

import skrobot
from skrobot.interfaces._pybullet import _check_available
from skrobot.interfaces._pybullet import PybulletRobotInterface


class TestPybulletRobotInterface(unittest.TestCase):

    @pytest.mark.skipif(_check_available() is False,
                        reason="Pybullet is not available")
    def test_init(self):
        fetch = skrobot.models.Fetch()
        PybulletRobotInterface(fetch, connect=2)
