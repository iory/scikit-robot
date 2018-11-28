import unittest

from numpy import deg2rad
from numpy import pi
from numpy import testing

from robot.coordinates import make_coords


class TestCoordinates(unittest.TestCase):

    def test_transform(self):
        coord = make_coords()
        coord.transform(make_coords(pos=[1, 2, 3]))
        testing.assert_array_equal(coord.pos,
                                   [1, 2, 3])

    def test_transformation(self):
        c = make_coords(rot=[deg2rad(10), 0, 0])
        d = make_coords(rot=[deg2rad(20), 0, 0])
        testing.assert_almost_equal(
            c.transformation(d).worldrot(),
            make_coords(rot=[deg2rad(10), 0, 0]).worldrot())

    def test_inverse_transformation(self):
        coord_a = make_coords(pos=[1, 1, 1])
        testing.assert_almost_equal(
            coord_a.inverse_transformation().worldpos(),
            [-1, -1, -1])

        pos = [0.13264493, 0.05263172, 0.93042636]
        q = [-0.20692513, 0.50841015, 0.82812527, 0.1136206]
        coord_b = make_coords(pos=pos, rot=q)
        testing.assert_almost_equal(
            coord_b.inverse_transformation().worldpos(),
            [-0.41549991, -0.12132025,  0.83588229])
        testing.assert_almost_equal(
            coord_b.inverse_transformation().quaternion,
            [0.20692513, 0.50841015, 0.82812527, 0.1136206])

    def rotate(self):
        c = make_coords(pos=[1, 2, 3])
        c.rotate(pi / 7.0, 'y', 'world')
        c.rotate(pi / 11.0, 'x', 'local')
        testing.assert_almost_equal(
            c.worldrot(),
            [[0.900969, 0.122239, 0.416308],
             [0.0, 0.959493, -0.281733],
             [-0.433884, 0.253832, 0.864473]])
