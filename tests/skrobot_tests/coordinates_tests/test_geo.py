import unittest

import numpy as np
from numpy import pi
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
        testing.assert_array_almost_equal(
            target_coords.rpy_angle()[0],
            [0, 0, -1.57079633])

        # case of rot_angle_cos == 1.0
        target_coords = make_coords(pos=[1.0, 1.0, 1.0])
        orient_coords_to_axis(target_coords, [0, 0, 1])

        testing.assert_array_equal(target_coords.worldpos(),
                                   [1, 1, 1])
        testing.assert_array_almost_equal(
            target_coords.rpy_angle()[0],
            [0, 0, 0])

        # case of rot_angle_cos == -1.0
        target_coords = make_coords()
        orient_coords_to_axis(target_coords, [0, 0, -1])

        testing.assert_array_equal(target_coords.worldpos(),
                                   [0, 0, 0])
        testing.assert_array_almost_equal(
            target_coords.rpy_angle()[0],
            [0, 0, pi])

    def test_rotate_points(self):
        points = np.array([1, 0, 0])
        rot_points = rotate_points(points, [1, 0, 0], [0, 0, 1])
        testing.assert_almost_equal(rot_points, [[0, 0, 1]])

        points = np.array([[1, 0, 0]])
        rot_points = rotate_points(points, [1, 0, 0], [0, 0, 1])
        testing.assert_almost_equal(rot_points, [[0, 0, 1]])

        points = np.array([[1, 0, 0]])
        rot_points = rotate_points(points, [0, 0, 1], [0, 0, 1])
        testing.assert_almost_equal(rot_points, [[1, 0, 0]])
