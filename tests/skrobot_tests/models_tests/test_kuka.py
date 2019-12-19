import unittest

import skrobot


class TestKuka(unittest.TestCase):

    def test_init(self):
        skrobot.models.Kuka()
