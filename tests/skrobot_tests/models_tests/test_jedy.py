import unittest

from skrobot.data import jedy_urdfpath


class TestJedy(unittest.TestCase):

    def test_light_and_draco_are_exclusive(self):
        with self.assertRaises(ValueError):
            jedy_urdfpath(light=True, draco=True)
