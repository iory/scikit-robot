import unittest

import robot


class TestPR2(unittest.TestCase):

    def test_init(self):
        pr2 = robot.robot_models.PR2()


class TestFetch(unittest.TestCase):

    def test_init(self):
        fetch = robot.robot_models.Fetch()


class TestKuka(unittest.TestCase):

    def test_init(self):
        kuka = robot.robot_models.Kuka()
