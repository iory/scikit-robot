import unittest

from skrobot.data import room13bc_urdfpath


class TestRoom13bcUrdfpath(unittest.TestCase):

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            room13bc_urdfpath(model='kitchen')

    def test_light_has_no_glb(self):
        with self.assertRaises(ValueError):
            room13bc_urdfpath(glb=True, light=True)
