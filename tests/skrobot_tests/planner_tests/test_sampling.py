import unittest

import numpy as np
from numpy import testing

from skrobot.planner.sampling import resample
from skrobot.planner.sampling import rrt_connect
from skrobot.planner.sampling import shortcut


def _wall_with_gap(q):
    """A 2-D world: a wall at x in [0.4, 0.6] with a gap at y > 1.5."""
    x, y = q
    return not (0.4 <= x <= 0.6 and y <= 1.5)


class TestRRTConnect(unittest.TestCase):

    def test_finds_a_way_through_the_gap(self):
        start = np.array([0.0, 0.0])
        goal = np.array([1.0, 0.0])
        lower = np.array([-0.5, -0.5])
        upper = np.array([1.5, 2.0])
        path = rrt_connect(start, goal, _wall_with_gap, lower, upper,
                           step=0.2, resolution=0.05, seed=1)
        self.assertIsNotNone(path)
        testing.assert_allclose(path[0], start)
        testing.assert_allclose(path[-1], goal)
        # Every configuration, and every step between two, is valid at
        # the resolution the search checked at. Sampling finer than that
        # is asking for a guarantee the algorithm does not make: a
        # violation narrower than one check spacing can pass between two
        # check points, and this world has a hard boundary to graze.
        for a, b in zip(path[:-1], path[1:]):
            n = max(int(np.ceil(np.max(np.abs(b - a)) / 0.05)), 1)
            for i in range(n + 1):
                self.assertTrue(_wall_with_gap(a + (b - a) * (i / float(n))))
        # It went through the gap, not the wall.
        self.assertGreater(max(q[1] for q in path), 1.5)

        # Shortcutting keeps the endpoints and validity and never lengthens.
        length = sum(np.max(np.abs(b - a)) for a, b in zip(path[:-1], path[1:]))
        short = shortcut(path, _wall_with_gap, resolution=0.05, seed=1)
        testing.assert_allclose(short[0], start)
        testing.assert_allclose(short[-1], goal)
        short_length = sum(np.max(np.abs(b - a))
                           for a, b in zip(short[:-1], short[1:]))
        self.assertLessEqual(short_length, length + 1e-9)
        for a, b in zip(short[:-1], short[1:]):
            n = max(int(np.ceil(np.max(np.abs(b - a)) / 0.05)), 1)
            for i in range(n + 1):
                self.assertTrue(_wall_with_gap(a + (b - a) * (i / float(n))))

    def test_gives_up_within_the_budget(self):
        # No gap: nothing can get across.
        def wall(q):
            return not (0.4 <= q[0] <= 0.6)
        path = rrt_connect(np.array([0.0, 0.0]), np.array([1.0, 0.0]), wall,
                           np.array([-0.5, -0.5]), np.array([1.5, 2.0]),
                           max_iterations=300, time_limit=2.0, seed=0)
        self.assertIsNone(path)


class TestResample(unittest.TestCase):

    def test_spreads_points_evenly_and_keeps_the_ends(self):
        path = [np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                np.array([1.0, 3.0])]
        out = resample(path, 5)
        self.assertEqual(out.shape, (5, 2))
        testing.assert_allclose(out[0], path[0])
        testing.assert_allclose(out[-1], path[-1])
        # Arc length (max-norm) is 4; points every 1.0 along it.
        testing.assert_allclose(out[1], [1.0, 0.0])
        testing.assert_allclose(out[2], [1.0, 1.0])
        testing.assert_allclose(out[3], [1.0, 2.0])
        steps = np.max(np.abs(np.diff(out, axis=0)), axis=1)
        testing.assert_allclose(steps, 1.0)


if __name__ == '__main__':
    unittest.main()
