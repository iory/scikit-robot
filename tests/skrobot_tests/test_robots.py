import unittest

import skrobot


class TestPR2(unittest.TestCase):

    def test_init(self):
        pr2 = skrobot.robot_models.PR2()


class TestFetch(unittest.TestCase):

    def test_init(self):
        fetch = skrobot.robot_models.Fetch()


class TestKuka(unittest.TestCase):

    def test_init(self):
        kuka = skrobot.robot_models.Kuka()
