import unittest

import skrobot


try:
    import fcl  # NOQA
    _fcl_available = True
except ImportError:
    _fcl_available = False


@unittest.skipUnless(_fcl_available, 'python-fcl is not installed')
class TestSelfCollisionCheck(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.kuka = skrobot.models.Kuka()

    def test_adjacent_links_are_ignored_by_default(self):
        robot = self.kuka
        adjacent = robot._adjacent_collision_link_pairs()
        _, every = robot.self_collision_check(ignore_adjacent=False)
        _, names = robot.self_collision_check()
        self.assertTrue(every & adjacent)
        self.assertFalse(names & adjacent)
        self.assertEqual(names, every - adjacent)

    def test_ignore_pairs(self):
        robot = self.kuka
        _, names = robot.self_collision_check()
        # Given in the reverse order on purpose: pairs match in any order.
        flipped = [(b, a) for a, b in names]
        is_collision, rest = robot.self_collision_check(ignore_pairs=flipped)
        self.assertFalse(is_collision)
        self.assertEqual(rest, set())
