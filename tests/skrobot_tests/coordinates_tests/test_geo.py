import unittest

import numpy as np
from numpy import testing

from skrobot.coordinates import make_coords
from skrobot.coordinates.geo import midcoords
from skrobot.coordinates.geo import orient_coords_to_axis
from skrobot.coordinates.geo import rotate_points
from skrobot.coordinates.math import matrix2quaternion


class TestGeo(unittest.TestCase):

    def test_midcoords(self):
        a = make_coords(pos=[1.0, 1.0, 1.0])
        b = make_coords()
        c = midcoords(0.5, a, b)
        testing.assert_array_equal(c.worldpos(),
                                   [0.5, 0.5, 0.5])
        testing.assert_array_equal(matrix2quaternion(c.worldrot()),
                                   [1, 0, 0, 0])

    def test_orient_coords_to_axis(self):
        target_coords = make_coords(pos=[1.0, 1.0, 1.0])
        orient_coords_to_axis(target_coords, [0, 1, 0])

        testing.assert_array_equal(target_coords.worldpos(),
                                   [1, 1, 1])
        from skrobot.coordinates.math import matrix2ypr
        testing.assert_array_almost_equal(
            matrix2ypr(target_coords.rotation),
            [0, 0, -1.57079633])

        # case of rot_angle_cos == 1.0
        target_coords = make_coords(pos=[1.0, 1.0, 1.0])
        orient_coords_to_axis(target_coords, [0, 0, 1])

        testing.assert_array_equal(target_coords.worldpos(),
                                   [1, 1, 1])
        testing.assert_array_almost_equal(
            matrix2ypr(target_coords.rotation),
            [0, 0, 0])

        # case of rot_angle_cos == -1.0
        # Any half turn about an axis perpendicular to z sends z to -z, so
        # assert where the axis ends up rather than picking one of them.
        target_coords = make_coords()
        orient_coords_to_axis(target_coords, [0, 0, -1])

        testing.assert_array_equal(target_coords.worldpos(),
                                   [0, 0, 0])
        testing.assert_array_almost_equal(
            target_coords.rotation[:, 2], [0, 0, -1])
        testing.assert_array_almost_equal(
            np.linalg.det(target_coords.rotation), 1.0)

    def test_rotate_points(self):
        points = np.array([1, 0, 0])
        rot_points = rotate_points(points, [1, 0, 0], [0, 0, 1])
        testing.assert_almost_equal(rot_points, [[0, 0, 1]])

    def test_rotate_points_parallel_and_anti_parallel(self):
        points = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])

        # a == b: nothing moves.
        testing.assert_almost_equal(
            rotate_points(points, [0, 0, 1], [0, 0, 1]), points)

        # a == -b: a half turn, which must keep the norms and be a rotation
        # rather than a point reflection (that would negate every component).
        rotated = rotate_points(points, [0, 0, 1], [0, 0, -1])
        testing.assert_almost_equal(
            np.linalg.norm(rotated, axis=1), np.linalg.norm(points, axis=1))
        testing.assert_almost_equal(rotated[:, 2], -points[:, 2])
        self.assertFalse(np.allclose(rotated, -points),
                         'anti-parallel case is a reflection, not a rotation')

        # The rotation must be the one that carries a onto b.
        for a, b in (([0, 0, 1], [0, 0, -1]), ([1, 0, 0], [-1, 0, 0]),
                     ([1, 1, 1], [-1, -1, -1])):
            a = np.array(a, dtype=np.float64)
            a /= np.linalg.norm(a)
            b = np.array(b, dtype=np.float64)
            b /= np.linalg.norm(b)
            testing.assert_almost_equal(rotate_points(a, a, b)[0], b)

        points = np.array([[1, 0, 0]])
        rot_points = rotate_points(points, [1, 0, 0], [0, 0, 1])
        testing.assert_almost_equal(rot_points, [[0, 0, 1]])

        points = np.array([[1, 0, 0]])
        rot_points = rotate_points(points, [0, 0, 1], [0, 0, 1])
        testing.assert_almost_equal(rot_points, [[1, 0, 0]])
