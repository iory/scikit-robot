import unittest

import robot


class TestPR2(unittest.TestCase):

    def test_init(self):
        pr2 = robot.robots.PR2()


class TestFetch(unittest.TestCase):

    def test_init(self):
        fetch = robot.robots.Fetch()


class TestKuka(unittest.TestCase):

    def test_init(self):
        kuka = robot.robots.Kuka()
