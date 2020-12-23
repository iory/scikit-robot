import unittest

import numpy as np
from numpy import deg2rad
from numpy import pi
from numpy import sign
from numpy import testing

from skrobot.coordinates import make_cascoords
from skrobot.coordinates import make_coords
from skrobot.coordinates.math import rotation_matrix
from skrobot.coordinates.math import rpy_matrix
from skrobot.coordinates import Transform


class testTransform(unittest.TestCase):

    def test__call__(self):
        # see test_transform_vector for these values
        pos = [0.13264493, 0.05263172, 0.93042636]
        q = [-0.20692513, 0.50841015, 0.82812527, 0.1136206]
        coords = make_coords(pos=pos, rot=q)
        tf = coords.get_transform()

        pt_original = np.array([0.2813606, 0.97762403, 0.83617263])
        pt_transformed = tf(pt_original)
        testing.assert_almost_equal(pt_transformed, np.array(
            [0.70004566, 1.05660075, 0.29465928]))

        tf(np.zeros((100, 3)))  # ok
        with self.assertRaises(AssertionError):
            tf(np.zeros((100, 100, 3)))  # ng

    def test__mull__(self):
        trans12 = np.array([0, 0, 1])
        rot12 = rpy_matrix(pi / 2.0, 0, 0)
        tf12 = Transform(trans12, rot12)

        trans23 = np.array([1, 0, 0])
        rot23 = rpy_matrix(0, 0, pi / 2.0)
        tf23 = Transform(trans23, rot23)

        tf13 = tf12 * tf23

        # from principle
        testing.assert_almost_equal(
            tf13.translation, rot23.dot(trans12) + trans23)
        testing.assert_almost_equal(tf13.rotation, rot23.dot(rot12))

    def test_get_inverse(self):
        # this also checks __mull__
        trans = np.array([1, 1, 1])
        rot = rpy_matrix(pi / 2.0, pi / 3.0, pi / 5.0)
        tf = Transform(trans, rot)
        tf_inv = tf.get_inverse()

        tf_id = tf * tf_inv
        testing.assert_almost_equal(tf_id.translation, np.zeros(3))


class TestCoordinates(unittest.TestCase):

    def test___init__(self):
        coord = make_coords(pos=[1, 1, 1])
        testing.assert_array_equal(coord.translation, [1, 1, 1])

        coord = make_coords(pos=[1, 0, 1], rot=[pi, 0, 0])
        testing.assert_array_equal(coord.translation, [1, 0, 1])
        testing.assert_almost_equal(coord.rotation,
                                    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        coord = make_coords([[-1, 0, 0, 1],
                             [0, -1, 0, 0],
                             [0, 0, 1, -1],
                             [0, 0, 0, 1]],
                            rot=[pi, 0, 0])
        testing.assert_array_equal(coord.translation, [1, 0, -1])
        testing.assert_almost_equal(coord.rotation,
                                    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    def test_transform(self):
        coord = make_coords()
        coord.transform(make_coords(pos=[1, 2, 3]))
        testing.assert_array_equal(coord.translation,
                                   [1, 2, 3])
        original_id = hex(id(coord))
        coord.transform(make_coords(pos=[1, 2, 3]),
                        coord)
        self.assertEqual(original_id,
                         hex(id(coord)))

        coord = make_coords().rotate(pi / 2.0, 'y')
        coord.transform(make_coords(pos=[1, 2, 3]),
                        'world')
        testing.assert_array_equal(coord.translation,
                                   [1, 2, 3])

        wrt = make_coords().rotate(pi / 2.0, 'y')
        coord = make_coords()
        coord.transform(make_coords(pos=[1, 2, 3]), wrt)
        testing.assert_almost_equal(coord.translation, [3.0, 2.0, -1.0])

    def test_move_coords(self):
        coord = make_coords()
        target_coord = make_coords(
            pos=(1, 2, 3), rot=(0, 0, 1, 0))
        local_coord = make_coords(
            pos=(-2, -2, -1))
        coord.move_coords(target_coord, local_coord)
        result = coord.copy_worldcoords().transform(local_coord)
        testing.assert_almost_equal(
            result.translation, (1, 2, 3))
        testing.assert_almost_equal(
            result.quaternion, (0, 0, 1, 0))

    def test_translate(self):
        c = make_coords()
        testing.assert_almost_equal(
            c.translation, [0, 0, 0])
        c.translate([0.1, 0.2, 0.3])
        testing.assert_almost_equal(
            c.translation, [0.1, 0.2, 0.3])

        c = make_coords().rotate(pi / 2.0, 'y')
        c.translate([0.1, 0.2, 0.3], 'local')
        testing.assert_almost_equal(
            c.translation, [0.3, 0.2, -0.1])

        c = make_coords().rotate(pi / 2.0, 'y')
        c.translate([0.1, 0.2, 0.3], 'world')
        testing.assert_almost_equal(
            c.translation, [0.1, 0.2, 0.3])

        c = make_coords()
        c2 = make_coords().translate([0.1, 0.2, 0.3])
        c.translate([0.1, 0.2, 0.3], c2)
        testing.assert_almost_equal(
            c.translation, [0.1, 0.2, 0.3])

        c = make_coords().rotate(pi / 3.0, 'z')
        c2 = make_coords().rotate(pi / 2.0, 'y')
        c.translate([0.1, 0.2, 0.3], c2)
        testing.assert_almost_equal(
            c.translation, [0.3, 0.2, -0.1])
        testing.assert_almost_equal(
            c.rpy_angle()[0], [pi / 3.0, 0, 0])

    def test_transform_vector(self):
        pos = [0.13264493, 0.05263172, 0.93042636]
        q = [-0.20692513, 0.50841015, 0.82812527, 0.1136206]
        coord = make_coords(pos=pos, rot=q)
        testing.assert_almost_equal(
            coord.transform_vector([0.2813606, 0.97762403, 0.83617263]),
            [0.70004566, 1.05660075, 0.29465928])

        coord = make_coords(pos=[0, 0, 1])
        testing.assert_almost_equal(
            coord.transform_vector([0, 0, 1]),
            [0, 0, 2])

        coord = make_coords(pos=[1, 1, 1])
        testing.assert_almost_equal(
            coord.transform_vector([-1, -1, -1]),
            [0, 0, 0])

        # batch transform
        coord = make_coords(pos=[1, 1, 1])
        coord.rotate(pi, 'z')
        testing.assert_almost_equal(
            coord.transform_vector(((0, 1, 2),
                                    (2, 3, 4),
                                    (5, 6, 7))),
            ((1, 0, 3),
             (-1, -2, 5),
             (-4, -5, 8)))

    def test_inverse_transform_vector(self):
        pos = [0.13264493, 0.05263172, 0.93042636]
        q = [-0.20692513, 0.50841015, 0.82812527, 0.1136206]
        coord = make_coords(pos=pos, rot=q)
        testing.assert_almost_equal(
            coord.inverse_transform_vector(
                [0.2813606, 0.97762403, 0.83617263]),
            [0.63310725, 0.55723807, 0.41865477])

        coord = make_coords(pos=[0, 0, 1])
        testing.assert_almost_equal(
            coord.inverse_transform_vector([0, 0, 1]),
            [0, 0, 0])

        coord = make_coords(pos=[1, 1, 1])
        testing.assert_almost_equal(
            coord.inverse_transform_vector([-1, -1, -1]),
            [-2, -2, -2])

        # batch transform
        coord = make_coords(pos=(1, 1, 1))
        coord.rotate(pi, 'z')
        testing.assert_almost_equal(
            coord.inverse_transform_vector(((0, 1, 2),
                                            (2, 3, 4),
                                            (5, 6, 7))),
            ((1, 0, 1),
             (-1, -2, 3),
             (-4, -5, 6)))

    def test_transformation(self):
        coord_a = make_coords(pos=[0, 0, 1])
        coord_b = make_coords(pos=[1, 0, 0])
        testing.assert_almost_equal(
            coord_a.transformation(coord_b).worldpos(),
            [1, 0, -1])
        testing.assert_almost_equal(
            coord_a.transformation(coord_b).quaternion,
            [1, 0, 0, 0])
        testing.assert_almost_equal(
            coord_b.transformation(coord_a).worldpos(),
            [-1, 0, 1])
        testing.assert_almost_equal(
            coord_b.transformation(coord_a).quaternion,
            [1, 0, 0, 0])

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
            [-0.41549991, -0.12132025, 0.83588229])
        testing.assert_almost_equal(
            coord_b.inverse_transformation().quaternion,
            [0.20692513, 0.50841015, 0.82812527, 0.1136206])

        # check inverse of transformation(worldcoords)
        testing.assert_almost_equal(
            coord_a.inverse_transformation().worldpos(),
            coord_a.transformation(make_coords()).worldpos())

    def rotate(self):
        c = make_coords(pos=[1, 2, 3])
        c.rotate(pi / 7.0, 'y', 'world')
        c.rotate(pi / 11.0, 'x', 'local')
        testing.assert_almost_equal(
            c.worldrot(),
            [[0.900969, 0.122239, 0.416308],
             [0.0, 0.959493, -0.281733],
             [-0.433884, 0.253832, 0.864473]])

    def test_quaternion(self):
        c = make_coords()
        testing.assert_almost_equal(
            c.quaternion, [1, 0, 0, 0])
        c.rotate(pi / 3, 'y').rotate(pi / 5, 'z')
        testing.assert_almost_equal(
            c.quaternion,
            [0.8236391, 0.1545085, 0.47552826, 0.26761657])

    def test_difference_position(self):
        coord1 = make_coords()
        coord2 = make_coords().translate([1, 2, 3])
        dif_pos = coord1.difference_position(coord2)
        testing.assert_almost_equal(dif_pos, [1, 2, 3])

        dif_pos = coord1.difference_position(coord2, translation_axis=False)
        testing.assert_almost_equal(dif_pos, [0, 0, 0])

        dif_pos = coord1.difference_position(coord2, translation_axis='x')
        testing.assert_almost_equal(dif_pos, [0, 2, 3])
        dif_pos = coord1.difference_position(coord2, translation_axis='y')
        testing.assert_almost_equal(dif_pos, [1, 0, 3])
        dif_pos = coord1.difference_position(coord2, translation_axis='z')
        testing.assert_almost_equal(dif_pos, [1, 2, 0])

        dif_pos = coord1.difference_position(coord2, translation_axis='xy')
        testing.assert_almost_equal(dif_pos, [0, 0, 3])
        dif_pos = coord1.difference_position(coord2, translation_axis='yz')
        testing.assert_almost_equal(dif_pos, [1, 0, 0])
        dif_pos = coord1.difference_position(coord2, translation_axis='zx')
        testing.assert_almost_equal(dif_pos, [0, 2, 0])

    def test_difference_rotation(self):
        coord1 = make_coords()
        coord2 = make_coords(rot=rpy_matrix(pi / 2.0, pi / 3.0, pi / 5.0))
        dif_rot = coord1.difference_rotation(coord2)
        testing.assert_almost_equal(dif_rot,
                                    [-0.32855112, 1.17434985, 1.05738936])
        dif_rot = coord1.difference_rotation(coord2, False)
        testing.assert_almost_equal(dif_rot,
                                    [0, 0, 0])

        dif_rot = coord1.difference_rotation(coord2, 'x')
        testing.assert_almost_equal(dif_rot,
                                    [0.0, 1.36034952, 0.78539816])
        dif_rot = coord1.difference_rotation(coord2, 'y')
        testing.assert_almost_equal(dif_rot,
                                    [0.35398131, 0.0, 0.97442695])
        dif_rot = coord1.difference_rotation(coord2, 'z')
        testing.assert_almost_equal(dif_rot,
                                    [-0.88435715, 0.74192175, 0.0])

        # TODO(iory) This case's rotation_axis='xx' is unstable
        # due to float point
        dif_rot = coord1.difference_rotation(coord2, 'xx')
        testing.assert_almost_equal(dif_rot[0], 0)
        testing.assert_almost_equal(abs(dif_rot[1]), 1.36034952)
        testing.assert_almost_equal(abs(dif_rot[2]), 0.78539816)
        testing.assert_almost_equal(sign(dif_rot[1]) * sign(dif_rot[2]), 1)

        dif_rot = coord1.difference_rotation(coord2, 'yy')
        testing.assert_almost_equal(
            dif_rot, [0.35398131, 0.0, 0.97442695])
        dif_rot = coord1.difference_rotation(coord2, 'zz')
        testing.assert_almost_equal(
            dif_rot, [-0.88435715, 0.74192175, 0.0])

        coord1 = make_coords()
        coord2 = make_coords().rotate(pi, 'x')
        dif_rot = coord1.difference_rotation(coord2, 'xm')
        testing.assert_almost_equal(dif_rot, [0, 0, 0])

        coord2 = make_coords().rotate(pi / 2.0, 'x')
        dif_rot = coord1.difference_rotation(coord2, 'xm')
        testing.assert_almost_equal(dif_rot, [-pi / 2.0, 0, 0])

        # corner case
        coord1 = make_coords()
        coord2 = make_coords().rotate(0.2564565431501872, 'y')
        dif_rot = coord1.difference_rotation(coord2, 'zy')
        testing.assert_almost_equal(dif_rot, [0, 0, 0])

        # norm == 0 case
        coord1 = make_coords()
        coord2 = make_coords()
        dif_rot = coord1.difference_rotation(coord2, 'xy')
        testing.assert_almost_equal(dif_rot, [0, 0, 0])

        coord1 = make_coords()
        coord2 = make_coords().rotate(pi / 2, 'x').rotate(pi / 2, 'y')
        dif_rot = coord1.difference_rotation(coord2, 'xy')
        testing.assert_almost_equal(dif_rot, [0, 0, pi / 2])
        dif_rot = coord1.difference_rotation(coord2, 'yx')
        testing.assert_almost_equal(dif_rot, [0, 0, pi / 2])

        coord1 = make_coords()
        coord2 = make_coords().rotate(pi / 2, 'y').rotate(pi / 2, 'z')
        dif_rot = coord1.difference_rotation(coord2, 'yz')
        testing.assert_almost_equal(dif_rot, [pi / 2, 0, 0])
        dif_rot = coord1.difference_rotation(coord2, 'zy')
        testing.assert_almost_equal(dif_rot, [pi / 2, 0, 0])

        coord1 = make_coords()
        coord2 = make_coords().rotate(pi / 2, 'z').rotate(pi / 2, 'x')
        dif_rot = coord1.difference_rotation(coord2, 'zx')
        testing.assert_almost_equal(dif_rot, [0, pi / 2, 0])
        dif_rot = coord1.difference_rotation(coord2, 'xz')
        testing.assert_almost_equal(dif_rot, [0, pi / 2, 0])


class TestCascadedCoordinates(unittest.TestCase):

    def test_changed(self):
        a = make_cascoords(rot=rotation_matrix(pi / 3, 'x'),
                           pos=[0.1, 0, 0])
        b = make_cascoords(rot=rotation_matrix(pi / 3, 'y'),
                           pos=[0.1, 0, 0.2])
        self.assertEqual(a._changed, True)
        self.assertEqual(b._changed, True)
        a.assoc(b)
        self.assertEqual(a._changed, False)
        self.assertEqual(b._changed, True)
        a.rotate(pi / 2.0, 'z')
        self.assertEqual(a._changed, True)
        self.assertEqual(b._changed, True)
        b.worldrot()
        self.assertEqual(a._changed, False)
        self.assertEqual(b._changed, False)

    def test_assoc(self):
        a = make_cascoords(rot=rotation_matrix(pi / 3, 'x'),
                           pos=[0.1, 0, 0],
                           name='a')
        b = make_cascoords(rot=rotation_matrix(pi / 3, 'y'),
                           pos=[0.1, 0, 0.2],
                           name='b')
        a.assoc(b)
        testing.assert_almost_equal(
            b.worldrot(),
            [[0.5, 0, 0.866025],
             [0, 1.0, 0.0],
             [-0.866025, 0, 0.5]],
            decimal=5)
        a.rotate(pi / 2.0, 'z')
        testing.assert_almost_equal(
            a.worldrot(),
            [[2.22044605e-16, -1.00000000e+00, 0.00000000e+00],
             [5.00000000e-01, 1.11022302e-16, -8.66025404e-01],
             [8.66025404e-01, 1.92296269e-16, 5.00000000e-01]],
            decimal=5)
        testing.assert_almost_equal(
            b.worldrot(),
            [[0.75, -0.5, -0.433013],
             [0.625, 0.75, 0.216506],
             [0.216506, -0.433013, 0.875]],
            decimal=5)
        testing.assert_almost_equal(
            a.worldpos(),
            [0.1, 0, 0])
        testing.assert_almost_equal(
            b.worldpos(),
            [-0.07320508, -0.08660254, 0.05])

        c = make_cascoords()
        with self.assertRaises(RuntimeError):
            c.assoc(b)

    def test_dissoc(self):
        a = make_cascoords(rot=rotation_matrix(pi / 3, 'x'),
                           pos=[0.1, 0, 0],
                           name='a')
        b = make_cascoords(rot=rotation_matrix(pi / 3, 'y'),
                           pos=[0.1, 0, 0.2],
                           name='b')
        a.assoc(b)
        a.dissoc(b)
        testing.assert_almost_equal(
            b.worldrot(),
            rotation_matrix(pi / 3, 'y'),
            decimal=5)
        a.rotate(pi / 2.0, 'z')
        testing.assert_almost_equal(
            a.worldrot(),
            [[2.22044605e-16, -1.00000000e+00, 0.00000000e+00],
             [5.00000000e-01, 1.11022302e-16, -8.66025404e-01],
             [8.66025404e-01, 1.92296269e-16, 5.00000000e-01]],
            decimal=5)
        testing.assert_almost_equal(
            b.worldrot(),
            rotation_matrix(pi / 3, 'y'),
            decimal=5)
        testing.assert_almost_equal(
            a.worldpos(),
            [0.1, 0, 0])
        testing.assert_almost_equal(
            b.worldpos(),
            [0.1, 0, 0.2])

    def test_worldcoords(self):
        a = make_cascoords(rot=rotation_matrix(pi / 3, 'x'),
                           pos=[0.1, 0, 0],
                           name='a')
        b = make_cascoords(rot=rotation_matrix(pi / 3, 'y'),
                           pos=[0.1, 0, 0.2],
                           name='b',
                           parent=a)
        original_id = hex(id(b.worldcoords()))
        a.rotate(pi / 2.0, 'z')
        self.assertEqual(original_id,
                         hex(id(b.worldcoords())))

    def test_transform(self):
        a = make_cascoords(rot=rotation_matrix(pi / 3, 'x'),
                           pos=[0.1, 0, 0])
        b = make_cascoords(rot=rotation_matrix(pi / 3, 'y'),
                           pos=[0.1, 0, 0.2])
        a.assoc(b)

        testing.assert_almost_equal(
            b.copy_worldcoords().transform(
                make_cascoords(pos=(-0.1, -0.2, -0.3)), 'local').translation,
            (-0.20980762113533155, -0.1999999999999999, 0.13660254037844383))

        testing.assert_almost_equal(
            b.copy_worldcoords().transform(
                make_cascoords(pos=(-0.1, -0.2, -0.3)), 'world').translation,
            (0, -0.2, -0.1))

        c = make_coords(pos=(-0.2, -0.3, -0.4)).rotate(pi / 2, 'y')
        b.transform(
            make_cascoords(pos=(-0.1, -0.2, -0.3)), c)
        testing.assert_almost_equal(
            b.translation, (-0.3, 0.15980762113533148, 0.32320508075688775))
        testing.assert_almost_equal(
            b.copy_worldcoords().translation,
            (-0.2, -0.2, 0.3))

        wrts = ['local', 'world', 'parent']
        for wrt in wrts:
            b.transform(
                make_cascoords(pos=(-0.1, -0.2, -0.3)), wrt=wrt)
