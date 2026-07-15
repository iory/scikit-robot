import pickle
import unittest
import warnings

import numpy as np
from numpy import deg2rad
from numpy import pi
from numpy import testing

from skrobot.coordinates import CascadedCoords
from skrobot.coordinates import make_cascoords
from skrobot.coordinates import make_coords
from skrobot.coordinates import Transform
from skrobot.coordinates.base import coordinates_distance
from skrobot.coordinates.base import lerp_coordinates
from skrobot.coordinates.base import slerp_coordinates
from skrobot.coordinates.base import wrt as wrt_vector
from skrobot.coordinates.math import _check_valid_rotation
from skrobot.coordinates.math import matrix2ypr
from skrobot.coordinates.math import quaternion2matrix
from skrobot.coordinates.math import rotation_distance
from skrobot.coordinates.math import rotation_matrix
from skrobot.coordinates.math import rotation_matrix_to_axis_angle_vector
from skrobot.coordinates.math import rpy_matrix
from skrobot.coordinates.math import wxyz2xyzw


def _homogeneous(coords):
    """Coordinates -> the 4x4 matrix it stands for."""
    matrix = np.eye(4)
    matrix[:3, :3] = coords.rotation
    matrix[:3, 3] = coords.translation
    return matrix


def _coords_pairs():
    """Deterministic, non-degenerate (a, b) pairs to check operations on.

    Drawn from a private RandomState so the set never depends on (or
    perturbs) the global numpy RNG.
    """
    rng = np.random.RandomState(0)
    pairs = []
    for _ in range(12):
        made = []
        for _ in range(2):
            axis = rng.randn(3)
            axis = axis / np.linalg.norm(axis)
            made.append(make_coords(
                pos=rng.uniform(-3, 3, 3),
                rot=rotation_matrix(rng.uniform(-pi, pi), axis)))
        pairs.append(tuple(made))
    return pairs


class TestTransform(unittest.TestCase):

    def test_transform_vector_and_rotate_vector(self):
        # see test_transform_vector for these values
        pos = [0.13264493, 0.05263172, 0.93042636]
        q = [-0.20692513, 0.50841015, 0.82812527, 0.1136206]
        coords = make_coords(pos=pos, rot=q)
        tf = coords.get_transform()

        pt_original = np.array([0.2813606, 0.97762403, 0.83617263])
        pt_transformed = tf.transform_vector(pt_original)
        pt_ground_truth = coords.transform_vector(pt_original)
        testing.assert_equal(pt_transformed, pt_ground_truth)

        pt_transformed = tf.rotate_vector(pt_original)
        pt_ground_truth = coords.rotate_vector(pt_original)
        testing.assert_equal(pt_transformed, pt_ground_truth)

        tf.transform_vector(np.zeros((100, 3)))  # ok
        with self.assertRaises(AssertionError):
            tf.transform_vector(np.zeros((100, 100, 3)))  # ng

        tf.rotate_vector(np.zeros((100, 3)))  # ok
        with self.assertRaises(AssertionError):
            tf.rotate_vector(np.zeros((100, 100, 3)))  # ng

    def test_transform_vector_values_1d_and_2d(self):
        # The existing batch checks pass np.zeros((100, 3)) and only assert
        # that nothing raises; all-zero input cannot tell + from -, nor a
        # rotation from its transpose. Pin actual values instead.
        translation = np.array([1.0, 2.0, 3.0])
        rotation = rotation_matrix(0.5, [0.3, -0.4, 0.5])
        tf = Transform(translation, rotation)
        points = np.array([[0.5, -1.5, 2.5], [0.0, 0.0, 0.0],
                           [-3.0, 1.0, 0.25]])

        batch = tf.transform_vector(points)
        self.assertEqual(batch.shape, points.shape)
        testing.assert_almost_equal(batch, points.dot(rotation.T) + translation)

        rotated = tf.rotate_vector(points)
        self.assertEqual(rotated.shape, points.shape)
        testing.assert_almost_equal(rotated, points.dot(rotation.T))
        # A zero vector is unmoved by a rotation.
        testing.assert_almost_equal(rotated[1], [0, 0, 0])

        # Each row must equal the 1-D call on that row.
        for i, point in enumerate(points):
            testing.assert_almost_equal(
                tf.transform_vector(point), rotation.dot(point) + translation)
            testing.assert_almost_equal(batch[i], tf.transform_vector(point))
            testing.assert_almost_equal(rotated[i], tf.rotate_vector(point))

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

    def test_inverse_transformation(self):
        # this also checks __mull__
        trans = np.array([1, 1, 1])
        rot = rpy_matrix(pi / 2.0, pi / 3.0, pi / 5.0)
        tf = Transform(trans, rot)
        tf_inv = tf.inverse_transformation()

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

    def test_init_rotation_forms(self):
        # rot accepts a 3x3 matrix, a quaternion or rpy angles, and pos can
        # carry a whole 4x4 matrix.
        quaternion = np.array([0.9238795, 0.3826834, 0.0, 0.0])
        want = quaternion2matrix(quaternion)

        testing.assert_almost_equal(make_coords(rot=np.eye(3)).rotation, np.eye(3))
        testing.assert_almost_equal(make_coords(rot=quaternion).rotation, want)
        testing.assert_almost_equal(
            make_coords(rot=[0.1, 0.2, 0.3]).rotation,
            rpy_matrix(0.1, 0.2, 0.3))

        matrix = np.eye(4)
        matrix[:3, :3] = want
        matrix[:3, 3] = [1, 2, 3]
        from_matrix = make_coords(matrix)
        testing.assert_almost_equal(from_matrix.rotation, want)
        testing.assert_almost_equal(from_matrix.translation, [1, 2, 3])

    def test_init_quaternion_order(self):
        # The same rotation, spelled both ways round.
        wxyz = np.array([0.9238795, 0.3826834, 0.0, 0.0])
        want = quaternion2matrix(wxyz)

        testing.assert_almost_equal(
            make_coords(rot=wxyz, input_quaternion_order='wxyz').rotation, want)
        testing.assert_almost_equal(
            make_coords(rot=wxyz2xyzw(wxyz),
                        input_quaternion_order='xyzw').rotation, want)
        # Reading xyzw as wxyz must not silently give the same answer.
        self.assertFalse(np.allclose(
            make_coords(rot=wxyz2xyzw(wxyz),
                        input_quaternion_order='wxyz').rotation, want))
        with self.assertRaises(ValueError):
            make_coords(rot=wxyz, input_quaternion_order='bogus')

    def test_pickling(self):
        coords = make_coords()
        coords.translate([1, 2, 3])
        coords_again = pickle.loads(pickle.dumps(coords))
        testing.assert_almost_equal(
            coords.translation, coords_again.translation)

    def test_x_axis(self):
        coord = make_coords()
        testing.assert_array_equal(coord.x_axis, [1, 0, 0])

        x_axis = coord.x_axis
        x_axis[0] = 10
        testing.assert_array_equal(coord.x_axis, [1, 0, 0])

    def test_y_axis(self):
        coord = make_coords()
        testing.assert_array_equal(coord.y_axis, [0, 1, 0])

        y_axis = coord.y_axis
        y_axis[0] = 10
        testing.assert_array_equal(coord.y_axis, [0, 1, 0])

    def test_z_axis(self):
        coord = make_coords()
        testing.assert_array_equal(coord.z_axis, [0, 0, 1])

        z_axis = coord.z_axis
        z_axis[0] = 10
        testing.assert_array_equal(coord.z_axis, [0, 0, 1])

    def test_newcoords_relative_coords(self):
        # Test Coordinates.newcoords with relative_coords parameter
        coord = make_coords(pos=[1, 0, 0])

        # Test relative_coords='world'
        new_coord = make_coords(pos=[2, 3, 4])
        coord.newcoords(new_coord, relative_coords='world')
        testing.assert_array_equal(coord.translation, [2, 3, 4])

        # Test relative_coords='local'
        coord = make_coords(pos=[1, 0, 0]).rotate(pi / 2.0, 'z')
        local_coord = make_coords(pos=[1, 0, 0])
        coord.newcoords(local_coord, relative_coords='local')
        # relative_coords='local' means the input is already in local frame
        # So it should directly set the local coordinates
        testing.assert_almost_equal(coord.translation, [1, 0, 0])

        # Test with Coordinates reference frame
        ref_coord = make_coords(pos=[5, 5, 5])
        coord = make_coords(pos=[1, 1, 1])
        new_coord = make_coords(pos=[2, 2, 2])
        coord.newcoords(new_coord, relative_coords=ref_coord)
        # This should apply the transformation: ref_coord * new_coord
        expected_pos = ref_coord.transform_vector([2, 2, 2])
        testing.assert_array_equal(coord.translation, expected_pos)

    def test_rotate_vector_batch_matches_single(self):
        # (N, 3) went through np.matmul(R, v) unbranched, which does not raise
        # for N == 3 -- it silently computed something else.
        c = make_coords(pos=[1, 2, 3], rot=rotation_matrix(0.5, [0.3, -0.4, 0.5]))
        points = np.array([[0.5, -1.5, 2.5], [0.0, 0.0, 0.0], [-3.0, 1.0, 0.25]])

        rotated = c.rotate_vector(points)
        self.assertEqual(rotated.shape, points.shape)
        # A zero vector stays zero under any rotation.
        testing.assert_almost_equal(rotated[1], [0, 0, 0])
        # Every row must equal the single-vector call.
        for i, point in enumerate(points):
            testing.assert_almost_equal(rotated[i], c.rotate_vector(point))
        testing.assert_almost_equal(rotated, points.dot(c.rotation.T))
        # Transform exposes the same API and was already correct.
        testing.assert_almost_equal(
            rotated, c.get_transform().rotate_vector(points))

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

    def test_orient_with_matrix(self):
        coords = make_coords()
        rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        coords.orient_with_matrix(rotation_matrix, wrt='local')
        coords.orient_with_matrix(rotation_matrix, wrt='world')
        wrt = make_coords().rotate(np.pi / 2.0, 'z')
        coords.orient_with_matrix(rotation_matrix, wrt=wrt)

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
            matrix2ypr(c.rotation), [pi / 3.0, 0, 0])

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
        # Default: constrain all axes
        dif_pos = coord1.difference_position(coord2)
        testing.assert_almost_equal(dif_pos, [1, 2, 3])

        # position_mask=False means no constraint (all zeros)
        dif_pos = coord1.difference_position(coord2, position_mask=False)
        testing.assert_almost_equal(dif_pos, [0, 0, 0])

        # position_mask='x' means constrain x only
        dif_pos = coord1.difference_position(coord2, position_mask='x')
        testing.assert_almost_equal(dif_pos, [1, 0, 0])
        dif_pos = coord1.difference_position(coord2, position_mask='y')
        testing.assert_almost_equal(dif_pos, [0, 2, 0])
        dif_pos = coord1.difference_position(coord2, position_mask='z')
        testing.assert_almost_equal(dif_pos, [0, 0, 3])

        # position_mask='xy' means constrain x and y
        dif_pos = coord1.difference_position(coord2, position_mask='xy')
        testing.assert_almost_equal(dif_pos, [1, 2, 0])
        dif_pos = coord1.difference_position(coord2, position_mask='yz')
        testing.assert_almost_equal(dif_pos, [0, 2, 3])
        dif_pos = coord1.difference_position(coord2, position_mask='zx')
        testing.assert_almost_equal(dif_pos, [1, 0, 3])

    def test_difference_rotation(self):
        coord1 = make_coords()
        coord2 = make_coords(rot=rpy_matrix(pi / 2.0, pi / 3.0, pi / 5.0))
        # Default: constrain all axes
        dif_rot = coord1.difference_rotation(coord2)
        testing.assert_almost_equal(dif_rot,
                                    [-0.32855112, 1.17434985, 1.05738936])

        # rotation_mask=False means no constraint (all zeros)
        dif_rot = coord1.difference_rotation(coord2, rotation_mask=False)
        testing.assert_almost_equal(dif_rot, [0, 0, 0])

        # rotation_mask='x' means constrain x only
        dif_rot = coord1.difference_rotation(coord2, rotation_mask='x')
        testing.assert_almost_equal(dif_rot[0], -0.32855112)
        testing.assert_almost_equal(dif_rot[1], 0)
        testing.assert_almost_equal(dif_rot[2], 0)

        dif_rot = coord1.difference_rotation(coord2, rotation_mask='yz')
        testing.assert_almost_equal(dif_rot[0], 0)
        testing.assert_almost_equal(dif_rot[1], 1.17434985)
        testing.assert_almost_equal(dif_rot[2], 1.05738936)

        # Test rotation_mirror
        coord1 = make_coords()
        coord2 = make_coords().rotate(pi, 'x')
        dif_rot = coord1.difference_rotation(coord2, rotation_mask=True,
                                             rotation_mirror='x')
        testing.assert_almost_equal(dif_rot, [0, 0, 0], decimal=5)

        coord2 = make_coords().rotate(pi / 2.0, 'x')
        dif_rot = coord1.difference_rotation(coord2, rotation_mask=True,
                                             rotation_mirror='x')
        testing.assert_almost_equal(dif_rot, [-pi / 2.0, 0, 0])

        # norm == 0 case
        coord1 = make_coords()
        coord2 = make_coords()
        dif_rot = coord1.difference_rotation(coord2, rotation_mask='xy')
        testing.assert_almost_equal(dif_rot, [0, 0, 0])

        # Test rotation_mask with partial axis constraints
        # rotation_mask='z' constrains only z axis
        coord1 = make_coords()
        coord2 = make_coords().rotate(pi / 2, 'x').rotate(pi / 2, 'y')
        dif_rot_full = coord1.difference_rotation(coord2)
        dif_rot = coord1.difference_rotation(coord2, rotation_mask='z')
        testing.assert_almost_equal(dif_rot, [0, 0, dif_rot_full[2]])

        # rotation_mask='x' constrains only x axis
        coord1 = make_coords()
        coord2 = make_coords().rotate(pi / 2, 'y').rotate(pi / 2, 'z')
        dif_rot_full = coord1.difference_rotation(coord2)
        dif_rot = coord1.difference_rotation(coord2, rotation_mask='x')
        testing.assert_almost_equal(dif_rot, [dif_rot_full[0], 0, 0])

        # rotation_mask='y' constrains only y axis
        coord1 = make_coords()
        coord2 = make_coords().rotate(pi / 2, 'z').rotate(pi / 2, 'x')
        dif_rot_full = coord1.difference_rotation(coord2)
        dif_rot = coord1.difference_rotation(coord2, rotation_mask='y')
        testing.assert_almost_equal(dif_rot, [0, dif_rot_full[1], 0])

    def test_rotation_matrix_alias(self):
        """Test that rotation_matrix is an alias for rotation."""
        coords = make_coords()
        testing.assert_array_equal(coords.rotation, coords.rotation_matrix)
        testing.assert_array_equal(coords.rotation_matrix, np.eye(3))

    def test_rotation_matrix_setter(self):
        """Test that rotation_matrix setter works."""
        coords = make_coords()
        rot_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        coords.rotation_matrix = rot_z
        testing.assert_array_almost_equal(coords.rotation, rot_z)
        rot_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        coords.rotation = rot_x
        testing.assert_array_almost_equal(coords.rotation_matrix, rot_x)

    def test_translation_vector_alias(self):
        """Test that translation_vector is an alias for translation."""
        coords = make_coords()
        testing.assert_array_equal(coords.translation, coords.translation_vector)
        testing.assert_array_equal(coords.translation_vector, np.zeros(3))

    def test_translation_vector_setter(self):
        """Test that translation_vector setter works."""
        coords = make_coords()
        vec1 = np.array([1.0, 2.0, 3.0])
        coords.translation_vector = vec1
        testing.assert_array_equal(coords.translation, vec1)
        vec2 = np.array([4.0, 5.0, 6.0])
        coords.translation = vec2
        testing.assert_array_equal(coords.translation_vector, vec2)

    def test_aliases_mixed_usage(self):
        """Test that old and new attribute names can be used interchangeably."""
        coords = make_coords()
        coords.translation = [1.0, 2.0, 3.0]
        testing.assert_array_equal(coords.translation_vector, [1.0, 2.0, 3.0])
        coords.rotation_matrix = np.eye(3)
        testing.assert_array_equal(coords.rotation, np.eye(3))

    def test_aliases_with_transformations(self):
        """Test aliases work with coordinate transformations."""
        coords = make_coords()
        coords.rotate(pi / 2, 'z')
        testing.assert_array_almost_equal(coords.rotation, coords.rotation_matrix)
        coords.translate([1.0, 2.0, 3.0])
        testing.assert_array_almost_equal(coords.translation, coords.translation_vector)

    def test_align_axis_to_direction_small_and_near_half_turn(self):
        """Directions close to the current axis, or close to its opposite."""
        axis_index = {'x': 0, 'y': 1, 'z': 2}
        for degree in (0.0, 0.001, 1.0, 3.0, 5.0, 5.7, 90.0,
                       175.0, 177.0, 179.0, 179.999, 180.0):
            theta = np.deg2rad(degree)
            direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
            for name, index in axis_index.items():
                c = make_coords()
                c.align_axis_to_direction(direction, axis=name, wrt='world')
                aligned = c.rotation[:, index]
                testing.assert_almost_equal(
                    aligned, direction, decimal=5,
                    err_msg='axis {} at {} deg -> {}'.format(
                        name, degree, aligned))
                _check_valid_rotation(c.rotation)

    def test_align_axis_to_direction(self):
        """Test align_axis_to_direction method."""
        # Default: align z-axis to x direction
        c = make_coords()
        c.align_axis_to_direction([1, 0, 0])
        testing.assert_almost_equal(c.z_axis, [1, 0, 0])

        # Align z-axis to y direction
        c = make_coords()
        c.align_axis_to_direction([0, 1, 0])
        testing.assert_almost_equal(c.z_axis, [0, 1, 0])

        # Align x-axis to z direction
        c = make_coords()
        c.align_axis_to_direction([0, 0, 1], axis='x')
        testing.assert_almost_equal(c.x_axis, [0, 0, 1])

        # Align y-axis to negative x direction
        c = make_coords()
        c.align_axis_to_direction([-1, 0, 0], axis='y')
        testing.assert_almost_equal(c.y_axis, [-1, 0, 0])

        # Case: already aligned (rot_angle_cos == 1.0)
        c = make_coords()
        c.align_axis_to_direction([0, 0, 1])
        testing.assert_almost_equal(c.z_axis, [0, 0, 1])
        testing.assert_almost_equal(c.rotation, np.eye(3))

        # Case: opposite direction (rot_angle_cos == -1.0)
        c = make_coords()
        c.align_axis_to_direction([0, 0, -1])
        testing.assert_almost_equal(c.z_axis, [0, 0, -1])

        # Test method chaining
        c = make_coords()
        result = c.align_axis_to_direction([1, 0, 0])
        self.assertEqual(id(c), id(result))

        # Test with non-unit vector (should normalize)
        c = make_coords()
        c.align_axis_to_direction([2, 0, 0])
        testing.assert_almost_equal(c.z_axis, [1, 0, 0])

        # Test wrt='local'
        c = make_coords().rotate(pi / 2, 'z')
        # Local x-axis is world y-axis after rotation
        c.align_axis_to_direction([1, 0, 0], wrt='local')
        # z-axis should now point to world y direction
        testing.assert_almost_equal(c.z_axis, [0, 1, 0])

        # Test wrt='world' (default)
        c = make_coords().rotate(pi / 2, 'z')
        c.align_axis_to_direction([1, 0, 0], wrt='world')
        testing.assert_almost_equal(c.z_axis, [1, 0, 0])

        # Test wrt with Coordinates
        ref = make_coords().rotate(pi / 2, 'z')
        c = make_coords()
        # ref's x-axis is world's y-axis
        c.align_axis_to_direction([1, 0, 0], wrt=ref)
        testing.assert_almost_equal(c.z_axis, [0, 1, 0])

    def test_slerp(self):
        """Test slerp method."""
        # Basic position interpolation
        c1 = make_coords()
        c2 = make_coords(pos=[1, 0, 0])
        c_mid = c1.slerp(c2, 0.5)
        testing.assert_almost_equal(c_mid.translation, [0.5, 0, 0])

        # Rotation interpolation with SLERP
        c1 = make_coords()
        c2 = make_coords().rotate(pi / 2, 'z')
        c_mid = c1.slerp(c2, 0.5)
        expected_angle = pi / 4
        testing.assert_almost_equal(
            matrix2ypr(c_mid.rotation)[0], expected_angle)

        # Test boundary values
        c_start = c1.slerp(c2, 0.0)
        testing.assert_almost_equal(c_start.rotation, c1.rotation)
        c_end = c1.slerp(c2, 1.0)
        testing.assert_almost_equal(c_end.rotation, c2.rotation)

        # Test that original coords are not modified
        c1 = make_coords(pos=[1, 2, 3])
        c2 = make_coords(pos=[4, 5, 6])
        c_mid = c1.slerp(c2, 0.5)
        testing.assert_almost_equal(c1.translation, [1, 2, 3])
        testing.assert_almost_equal(c2.translation, [4, 5, 6])

    def test_lerp(self):
        """Test lerp method."""
        # Basic position interpolation
        c1 = make_coords()
        c2 = make_coords(pos=[2, 2, 2])
        c_mid = c1.lerp(c2, 0.5)
        testing.assert_almost_equal(c_mid.translation, [1, 1, 1])

        # Quarter interpolation
        c_quarter = c1.lerp(c2, 0.25)
        testing.assert_almost_equal(c_quarter.translation, [0.5, 0.5, 0.5])

        # Test boundary values
        c_start = c1.lerp(c2, 0.0)
        testing.assert_almost_equal(c_start.translation, [0, 0, 0])
        c_end = c1.lerp(c2, 1.0)
        testing.assert_almost_equal(c_end.translation, [2, 2, 2])

        # Test that original coords are not modified
        c1 = make_coords(pos=[1, 2, 3])
        c2 = make_coords(pos=[4, 5, 6])
        c_mid = c1.lerp(c2, 0.5)
        testing.assert_almost_equal(c1.translation, [1, 2, 3])
        testing.assert_almost_equal(c2.translation, [4, 5, 6])

    def test_interpolate(self):
        """Test interpolate method (alias for slerp)."""
        # Basic interpolation of position
        c1 = make_coords()
        c2 = make_coords(pos=[1, 0, 0])
        c_mid = c1.interpolate(c2, 0.5)
        testing.assert_almost_equal(c_mid.translation, [0.5, 0, 0])

        # Verify interpolate gives same result as slerp
        c1 = make_coords()
        c2 = make_coords(pos=[1, 1, 1]).rotate(pi / 3, 'y')
        c_interp = c1.interpolate(c2, 0.5)
        c_slerp = c1.slerp(c2, 0.5)
        testing.assert_almost_equal(c_interp.translation, c_slerp.translation)
        testing.assert_almost_equal(c_interp.rotation, c_slerp.rotation)

        # Test rotation interpolation
        c1 = make_coords()
        c2 = make_coords().rotate(pi / 2, 'z')
        c_mid = c1.interpolate(c2, 0.5)
        expected_angle = pi / 4
        testing.assert_almost_equal(
            matrix2ypr(c_mid.rotation)[0], expected_angle)

        # Test that original coords are not modified
        c1 = make_coords(pos=[1, 2, 3])
        c2 = make_coords(pos=[4, 5, 6])
        c_mid = c1.interpolate(c2, 0.5)
        testing.assert_almost_equal(c1.translation, [1, 2, 3])
        testing.assert_almost_equal(c2.translation, [4, 5, 6])
        testing.assert_almost_equal(c_mid.translation, [2.5, 3.5, 4.5])

    def test_transformation_wrt_variants(self):
        # transformation() branches on wrt; cover every accepted spelling.
        for a, b in _coords_pairs():
            ha, hb = _homogeneous(a), _homogeneous(b)
            local = np.linalg.inv(ha).dot(hb)
            world = hb.dot(np.linalg.inv(ha))

            testing.assert_almost_equal(
                _homogeneous(a.transformation(b, 'local')), local)
            # A Coordinates with no parent treats 'parent' as 'world'.
            testing.assert_almost_equal(
                _homogeneous(a.transformation(b, 'world')), world)
            testing.assert_almost_equal(
                _homogeneous(a.transformation(b, 'parent')), world)
            # Passing self is the same as 'local'.
            testing.assert_almost_equal(
                _homogeneous(a.transformation(b, a)), local)
            with self.assertRaises(ValueError):
                a.transformation(b, 'nonsense')

    def test_newcoords_with_explicit_pos(self):
        # newcoords(c, pos) takes c as a bare rotation and pos separately;
        # nothing exercised that path.
        rotation = rotation_matrix(0.3, 'z')
        position = np.array([1.0, 2.0, 3.0])

        c = make_coords()
        c.newcoords(rotation, position)
        testing.assert_almost_equal(c.rotation, rotation)
        testing.assert_almost_equal(c.translation, position)

        # A reference frame composes with both parts.
        reference = make_coords(pos=[1, 0, 0])
        c = make_coords(pos=[5, 5, 5])
        c.newcoords(rotation, position, relative_coords=reference)
        want = _homogeneous(reference).dot(
            _homogeneous(make_coords(pos=position, rot=rotation)))
        testing.assert_almost_equal(_homogeneous(c), want)

        # The stored arrays must be copies of the caller's.
        rotation = rotation_matrix(0.3, 'z')
        position = np.array([1.0, 2.0, 3.0])
        c = make_coords()
        c.newcoords(rotation, position)
        rotation[0, 0] = 99.0
        position[0] = 99.0
        testing.assert_almost_equal(c.rotation, rotation_matrix(0.3, 'z'))
        testing.assert_almost_equal(c.translation, [1.0, 2.0, 3.0])

    def test_newcoords_rejects_unknown_relative_coords(self):
        for bad in ('bogus', 'localish', ''):
            c = make_coords()
            with self.assertRaises(ValueError):
                c.newcoords(make_coords(), relative_coords=bad)

    def test_newcoords_relative_coords_variants(self):
        for a, b in _coords_pairs():
            hb = _homogeneous(b)

            # local / world / None all set the pose outright with no parent.
            for relative in ('local', 'world', None):
                c = a.copy_worldcoords()
                c.newcoords(b, relative_coords=relative)
                testing.assert_almost_equal(_homogeneous(c), hb)

            # Spelling is case-insensitive.
            c = a.copy_worldcoords()
            c.newcoords(b, relative_coords='LOCAL')
            testing.assert_almost_equal(_homogeneous(c), hb)

            # A Coordinates reference frame composes: ref * target.
            ref = a.copy_worldcoords()
            c = make_coords()
            c.newcoords(b, relative_coords=ref)
            testing.assert_almost_equal(
                _homogeneous(c), _homogeneous(ref).dot(hb))

            # 'parent' without a parent is an error, not a silent fallback.
            c = a.copy_worldcoords()
            with self.assertRaises(ValueError):
                c.newcoords(b, relative_coords='parent')

    def test_rotate_axis_forms(self):
        # rotate() accepts an axis name or a vector; they must agree.
        for a, _ in _coords_pairs():
            for name, vector in (('x', [1, 0, 0]), ('y', [0, 1, 0]),
                                 ('z', [0, 0, 1])):
                by_name = a.copy_worldcoords().rotate(0.6, name, 'local')
                by_vec = a.copy_worldcoords().rotate(0.6, vector, 'local')
                by_arr = a.copy_worldcoords().rotate(
                    0.6, np.array(vector, dtype=np.float64), 'local')
                testing.assert_almost_equal(
                    _homogeneous(by_name), _homogeneous(by_vec))
                testing.assert_almost_equal(
                    _homogeneous(by_name), _homogeneous(by_arr))

            # A non-unit axis is normalized, not taken at face value.
            testing.assert_almost_equal(
                _homogeneous(a.copy_worldcoords().rotate(0.6, [0, 0, 5])),
                _homogeneous(a.copy_worldcoords().rotate(0.6, [0, 0, 1])))

            # Without an axis, rotate() reads theta as a matrix, so a scalar
            # theta is rejected rather than silently ignored.
            for missing_axis in (None, False):
                with self.assertRaises(ValueError):
                    a.copy_worldcoords().rotate(0.6, missing_axis)

    def test_transform_vector_batch(self):
        for a, _ in _coords_pairs():
            ha = _homogeneous(a.copy_worldcoords())
            points = np.array([[0.5, -1.5, 2.5], [0.0, 0.0, 0.0],
                               [-3.0, 1.0, 0.25]])
            # (N, 3) must equal applying the single-vector form row by row.
            testing.assert_almost_equal(
                a.transform_vector(points),
                points.dot(ha[:3, :3].T) + ha[:3, 3])
            testing.assert_almost_equal(
                a.rotate_vector(points), points.dot(ha[:3, :3].T))
            for point in points:
                testing.assert_almost_equal(
                    a.transform_vector(point),
                    ha[:3, :3].dot(point) + ha[:3, 3])

    def test_rotation_and_translation_setters_copy(self):
        # The setters must store a copy; otherwise callers alias the internals.
        rot = rotation_matrix(0.3, [0, 0, 1])
        trans = np.array([1.0, 2.0, 3.0])
        c = make_coords()
        c.rotation = rot
        c.translation = trans
        rot[0, 0] = 99.0
        trans[0] = 99.0
        testing.assert_almost_equal(c.rotation, rotation_matrix(0.3, [0, 0, 1]))
        testing.assert_almost_equal(c.translation, [1.0, 2.0, 3.0])

    def test_matches_homogeneous_matrix_algebra(self):
        # A Coordinates is a rigid transform, so pin every operation against
        # plain 4x4 matrices rather than against skrobot itself.
        for a, b in _coords_pairs():
            ha, hb = _homogeneous(a), _homogeneous(b)

            got = a.copy_worldcoords().transform(b, 'local')
            testing.assert_almost_equal(_homogeneous(got), ha.dot(hb))

            got = a.copy_worldcoords().transform(b, 'world')
            testing.assert_almost_equal(_homogeneous(got), hb.dot(ha))

            got = a.transformation(b, 'local')
            testing.assert_almost_equal(
                _homogeneous(got), np.linalg.inv(ha).dot(hb))

            testing.assert_almost_equal(
                _homogeneous(a.inverse_transformation()), np.linalg.inv(ha))

            # The relative transform must land exactly on the target.
            testing.assert_almost_equal(
                _homogeneous(a.copy_worldcoords().transform(
                    a.transformation(b, 'local'), 'local')), hb)

            testing.assert_almost_equal(
                _homogeneous(a.copy_worldcoords().transform(
                    a.inverse_transformation(), 'local')), np.eye(4))

    def test_rotate_and_translate_wrt(self):
        for a, _ in _coords_pairs():
            ha = _homogeneous(a)
            for theta, axis in ((0.7, [1, 0, 0]), (-1.3, [0, 1, 0]),
                                (2.1, [0, 0, 1]), (0.4, [1, -2, 3])):
                hr = np.eye(4)
                hr[:3, :3] = rotation_matrix(theta, axis)

                # local: rotate about self's own axes
                testing.assert_almost_equal(
                    _homogeneous(a.copy_worldcoords().rotate(
                        theta, axis, 'local')), ha.dot(hr))
                # rotate_with_matrix must agree with rotate
                testing.assert_almost_equal(
                    _homogeneous(a.copy_worldcoords().rotate_with_matrix(
                        hr[:3, :3], 'local')),
                    _homogeneous(a.copy_worldcoords().rotate(
                        theta, axis, 'local')))

                # world: rotate about the world axes, origin stays put
                want = hr.dot(ha)
                want[:3, 3] = ha[:3, 3]
                testing.assert_almost_equal(
                    _homogeneous(a.copy_worldcoords().rotate(
                        theta, axis, 'world')), want)

            vec = np.array([0.3, -1.2, 2.0])
            ht = np.eye(4)
            ht[:3, 3] = vec
            testing.assert_almost_equal(
                _homogeneous(a.copy_worldcoords().translate(vec, 'local')),
                ha.dot(ht))
            want = ha.copy()
            want[:3, 3] = ha[:3, 3] + vec
            testing.assert_almost_equal(
                _homogeneous(a.copy_worldcoords().translate(vec, 'world')),
                want)

    def test_newcoords_replaces_pose_without_aliasing(self):
        for a, b in _coords_pairs():
            hb = _homogeneous(b)
            c = a.copy_worldcoords()
            c.newcoords(b)
            testing.assert_almost_equal(_homogeneous(c), hb)
            # Mutating the source afterwards must not reach into c.
            b.translate([1.0, 1.0, 1.0], 'world')
            b.rotate(0.5, 'z', 'world')
            testing.assert_almost_equal(_homogeneous(c), hb)

    def test_homogeneous_and_vector_helpers(self):
        for a, _ in _coords_pairs():
            ha = _homogeneous(a)
            testing.assert_almost_equal(a.T(), ha)
            testing.assert_almost_equal(
                _homogeneous(a ** -1), np.linalg.inv(ha))
            with self.assertRaises(NotImplementedError):
                a ** 2

            v = np.array([0.5, -1.5, 2.5])
            testing.assert_almost_equal(a.rotate_vector(v), ha[:3, :3].dot(v))
            testing.assert_almost_equal(
                a.inverse_rotate_vector(v), ha[:3, :3].T.dot(v))
            testing.assert_almost_equal(
                a.transform_vector(v), ha[:3, :3].dot(v) + ha[:3, 3])
            testing.assert_almost_equal(
                wrt_vector(a, v), a.transform_vector(v))

            # rpy_angle must rebuild the rotation it came from.
            rpy, _ = a.rpy_angle()
            testing.assert_almost_equal(
                rpy_matrix(rpy[0], rpy[1], rpy[2]), ha[:3, :3])

    def test_difference_position_and_rotation(self):
        for a, b in _coords_pairs():
            ha, hb = _homogeneous(a), _homogeneous(b)
            # difference_position is expressed in a's own frame.
            testing.assert_almost_equal(
                a.difference_position(b),
                ha[:3, :3].T.dot(hb[:3, 3] - ha[:3, 3]))
            # difference_rotation is the axis-angle taking a onto b.
            want = np.linalg.norm(rotation_matrix_to_axis_angle_vector(
                ha[:3, :3].T.dot(hb[:3, :3])))
            testing.assert_almost_equal(
                np.linalg.norm(a.difference_rotation(b)), want)
            testing.assert_almost_equal(
                a.difference_rotation(a), np.zeros(3), decimal=5)

    def test_lerp_and_slerp_coordinates(self):
        for a, b in _coords_pairs():
            ha, hb = _homogeneous(a), _homogeneous(b)
            for fn in (lerp_coordinates, slerp_coordinates):
                testing.assert_almost_equal(_homogeneous(fn(a, b, 0.0)), ha)
                testing.assert_almost_equal(_homogeneous(fn(a, b, 1.0)), hb)
                mid = fn(a, b, 0.5)
                # Translation is linear for both.
                testing.assert_almost_equal(
                    mid.translation, (ha[:3, 3] + hb[:3, 3]) / 2.0)
                _check_valid_rotation(mid.rotation)

    def test_coordinates_distance(self):
        for a, b in _coords_pairs():
            ha, hb = _homogeneous(a), _homogeneous(b)
            dist = coordinates_distance(a, b)
            testing.assert_almost_equal(
                dist[0], np.linalg.norm(ha[:3, 3] - hb[:3, 3]))
            testing.assert_almost_equal(
                dist[1], rotation_distance(ha[:3, :3], hb[:3, :3]))
            testing.assert_almost_equal(
                np.asarray(coordinates_distance(a, a)), np.zeros(2), decimal=6)


class TestCascadedCoordinates(unittest.TestCase):

    def test_pickling(self):
        a = make_cascoords(pos=[0.1, 0, 0])
        b = make_cascoords(pos=[0, 0, 0.1])
        c = make_cascoords(pos=[0, 0, -0.1])
        a.assoc(b, relative_coords="local")
        a.assoc(c, relative_coords="local")

        a_again = pickle.loads(pickle.dumps(a))

        # see __getstate__ and __setstate__ implementation
        assert a._worldcoords._hook == a.update  # original must be unchanged
        assert a_again._worldcoords._hook == a_again.update

        # test properly dumped and loaded
        assert len(a_again.descendants) == 2
        b_again = a_again.descendants[0]
        c_again = a_again.descendants[1]
        assert id(b_again.parent) == id(a_again)
        assert id(c_again.parent) == id(a_again)
        testing.assert_almost_equal(
            a_again.translation, a.translation)
        testing.assert_almost_equal(
            b_again.translation, b.translation)
        testing.assert_almost_equal(
            c_again.translation, c.translation)

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
        child = a.assoc(b)
        self.assertEqual(b, child)
        child = a.assoc(b, force=True)
        self.assertEqual(b, child)
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

        with self.assertRaises(TypeError):
            c.assoc(b, relative_coords=1, force=True)

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

    def test_orient_with_matrix(self):
        coords = make_cascoords()
        rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        coords.orient_with_matrix(rotation_matrix, wrt='local')
        coords.orient_with_matrix(rotation_matrix, wrt='parent')
        coords.orient_with_matrix(rotation_matrix, wrt='world')
        wrt = make_coords().rotate(np.pi / 2.0, 'z')
        coords.orient_with_matrix(rotation_matrix, wrt=wrt)

    def test_newcoords_cascaded_relative_coords(self):
        # Test CascadedCoords.newcoords with relative_coords parameter

        # Create parent and child coordinates
        parent = make_cascoords(pos=[1, 0, 0])
        child = make_cascoords(pos=[0, 1, 0])
        parent.assoc(child)

        # Test default behavior (relative_coords='local')
        new_local_coord = make_cascoords(pos=[5, 5, 5])
        child.newcoords(new_local_coord)
        # Child's local position should be [5, 5, 5]
        testing.assert_array_equal(child.translation, [5, 5, 5])

        # Test relative_coords='world'
        parent = make_cascoords(pos=[1, 0, 0])
        child = make_cascoords(pos=[0, 1, 0])
        parent.assoc(child)

        new_world_coord = make_cascoords(pos=[5, 5, 5])
        child.newcoords(new_world_coord, relative_coords='world')
        # Child's world position should be [5, 5, 5]
        testing.assert_array_equal(child.worldpos(), [5, 5, 5])

        # Test relative_coords='local'
        parent = make_cascoords(pos=[2, 0, 0])
        child = make_cascoords(pos=[0, 2, 0])
        parent.assoc(child)

        local_coord = make_cascoords(pos=[1, 1, 1])
        child.newcoords(local_coord, relative_coords='local')
        # Child's local translation should be [1, 1, 1]
        testing.assert_array_equal(child.translation, [1, 1, 1])

        # Test relative_coords='parent'
        parent = make_cascoords(pos=[3, 0, 0])
        child = make_cascoords(pos=[0, 3, 0])
        parent.assoc(child)

        parent_rel_coord = make_cascoords(pos=[2, 2, 2])
        child.newcoords(parent_rel_coord, relative_coords='parent')
        # Child's translation relative to parent should be [2, 2, 2]
        testing.assert_array_equal(child.translation, [2, 2, 2])

    def test_newcoords_with_parent_and_explicit_pos(self):
        # The parent branch of CascadedCoords.newcoords, with pos given
        # separately, is a different code path from the pos=None one.
        rotation = rotation_matrix(0.6, 'z')
        position = np.array([2.0, 2.0, 2.0])

        def attached():
            parent = make_cascoords(pos=[1, 0, 0],
                                    rot=rotation_matrix(0.4, 'y'))
            child = make_cascoords(pos=[0, 1, 0])
            parent.assoc(child, relative_coords='local')
            return parent, child

        # 'world' places the child at the target in world coordinates.
        parent, child = attached()
        child.newcoords(rotation, position.copy(), relative_coords='world')
        testing.assert_almost_equal(
            child.worldcoords().translation, position)
        testing.assert_almost_equal(child.worldcoords().rotation, rotation)

        # 'local' and 'parent' both mean parent-relative, so the world pose
        # picks up the parent's transform.
        want = _homogeneous(make_cascoords(
            pos=[1, 0, 0], rot=rotation_matrix(0.4, 'y'))).dot(
                _homogeneous(make_coords(pos=position, rot=rotation)))
        for relative in ('local', 'parent'):
            parent, child = attached()
            child.newcoords(rotation, position.copy(),
                            relative_coords=relative)
            testing.assert_almost_equal(child.translation, position)
            testing.assert_almost_equal(
                _homogeneous(child.worldcoords()), want,
                err_msg='relative_coords={}'.format(relative))

        # A Coordinates frame composes with the target.
        parent, child = attached()
        reference = make_coords(pos=[3, 0, 0])
        child.newcoords(rotation, position.copy(), relative_coords=reference)
        testing.assert_almost_equal(
            _homogeneous(make_coords(pos=child.translation,
                                     rot=child.rotation)),
            _homogeneous(reference.transformation(
                make_coords(pos=position, rot=rotation))))

        parent, child = attached()
        with self.assertRaises(ValueError):
            child.newcoords(rotation, position.copy(),
                            relative_coords='bogus')

    def test_newcoords_relative_coords_with_parent(self):
        # With a parent attached, only 'world' reinterprets the target;
        # 'local', 'parent' and None all set the parent-relative pose.
        target = make_cascoords(pos=[2, 2, 2])

        for relative in ('local', 'parent', None, 'LOCAL'):
            parent = make_cascoords(pos=[1, 0, 0])
            child = make_cascoords(pos=[0, 1, 0])
            parent.assoc(child, relative_coords='local')
            child.newcoords(target, relative_coords=relative)
            testing.assert_almost_equal(
                child.translation, [2, 2, 2],
                err_msg='relative_coords={}'.format(relative))
            testing.assert_almost_equal(
                child.worldcoords().translation, [3, 2, 2],
                err_msg='relative_coords={}'.format(relative))

        # 'world' places the child at the target in world coordinates, so its
        # stored pose becomes parent^-1 * target.
        parent = make_cascoords(pos=[1, 0, 0])
        child = make_cascoords(pos=[0, 1, 0])
        parent.assoc(child, relative_coords='local')
        child.newcoords(target, relative_coords='world')
        testing.assert_almost_equal(child.worldcoords().translation, [2, 2, 2])
        testing.assert_almost_equal(child.translation, [1, 2, 2])

    def test_rotate_fast_path_matches_fallback(self):
        # CascadedCoords.rotate inlines rotate_with_matrix + newcoords +
        # changed when wrt is 'local', there is no hook and nothing in the
        # chain is overridden. Every guard must fall back to the same answer.
        setup = dict(pos=[1, 2, 3],
                     rot=rotation_matrix(0.4, [0.3, -0.5, 0.2]))
        axis = np.array([0.0, 0.0, 1.0])

        fast = CascadedCoords(**setup)
        fast.rotate(0.7, axis, 'local')
        want = _homogeneous(fast.worldcoords())

        # A hook disables the fast path, and must still be called.
        calls = []
        hooked = CascadedCoords(hook=lambda: calls.append(1), **setup)
        hooked.rotate(0.7, axis, 'local')
        testing.assert_almost_equal(_homogeneous(hooked.worldcoords()), want)
        self.assertTrue(calls)

        # So does a subclass overriding any member of the chain.
        class OverridesNewcoords(CascadedCoords):
            def newcoords(self, *args, **kwargs):
                return super(OverridesNewcoords, self).newcoords(
                    *args, **kwargs)

        subclassed = OverridesNewcoords(**setup)
        subclassed.rotate(0.7, axis, 'local')
        testing.assert_almost_equal(
            _homogeneous(subclassed.worldcoords()), want)

        # A named axis never takes the fast path but means the same rotation.
        named = CascadedCoords(**setup)
        named.rotate(0.7, 'z', 'local')
        testing.assert_almost_equal(_homogeneous(named.worldcoords()), want)

        # The fast path must still mark descendants as changed.
        parent = CascadedCoords(**setup)
        child = CascadedCoords(pos=[0, 1, 0])
        parent.assoc(child, relative_coords='local')
        child.worldcoords()
        parent._changed = False
        child._changed = False
        parent.rotate(0.3, axis, 'local')
        self.assertTrue(child._changed)

    def test_check_validity_rejects_non_rotation(self):
        not_a_rotation = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 2.0]])
        with self.assertRaises(ValueError):
            make_coords(rot=not_a_rotation)
        with self.assertRaises(ValueError):
            make_coords(rot=not_a_rotation, check_validity=True)
        # check_validity=False is the escape hatch and stores it as given.
        loose = make_coords(rot=not_a_rotation, check_validity=False)
        testing.assert_almost_equal(loose.rotation, not_a_rotation)

    def test_rotate_with_matrix_matches_coordinates(self):
        # CascadedCoords overrides rotate/rotate_with_matrix; an unattached
        # one must still behave exactly like a plain Coordinates.
        for a, _ in _coords_pairs():
            for theta, axis in ((0.7, [1, 0, 0]), (-1.3, [0, 1, 0]),
                                (2.1, [1, -2, 3])):
                matrix = rotation_matrix(theta, axis)
                for wrt_frame in ('local', 'world'):
                    plain = make_coords(pos=a.translation, rot=a.rotation)
                    casc = make_cascoords(pos=a.translation, rot=a.rotation)
                    plain.rotate_with_matrix(matrix, wrt_frame)
                    casc.rotate_with_matrix(matrix, wrt_frame)
                    testing.assert_almost_equal(
                        _homogeneous(casc.worldcoords()), _homogeneous(plain),
                        err_msg='wrt={}'.format(wrt_frame))
                    # and it must agree with rotate() for the same rotation
                    other = make_cascoords(pos=a.translation, rot=a.rotation)
                    other.rotate(theta, axis, wrt_frame)
                    testing.assert_almost_equal(
                        _homogeneous(casc.worldcoords()),
                        _homogeneous(other.worldcoords()))

    def test_rotate_wrt_parent_and_world(self):
        for a, b in _coords_pairs():
            hp, hc = _homogeneous(a), _homogeneous(b)
            theta, axis = 0.9, np.array([1.0, -2.0, 0.5])
            axis = axis / np.linalg.norm(axis)
            hr = np.eye(4)
            hr[:3, :3] = rotation_matrix(theta, axis)

            # 'local' turns the child about its own axes.
            parent = make_cascoords(pos=a.translation, rot=a.rotation)
            child = make_cascoords(pos=b.translation, rot=b.rotation)
            parent.assoc(child, relative_coords='local')
            child.rotate(theta, axis, 'local')
            testing.assert_almost_equal(
                _homogeneous(child.worldcoords()), hp.dot(hc).dot(hr))

            # 'parent' turns it about the parent's axes, origin kept.
            parent = make_cascoords(pos=a.translation, rot=a.rotation)
            child = make_cascoords(pos=b.translation, rot=b.rotation)
            parent.assoc(child, relative_coords='local')
            child.rotate(theta, axis, 'parent')
            want = hr.dot(hc)
            want[:3, 3] = hc[:3, 3]
            testing.assert_almost_equal(
                _homogeneous(child.worldcoords()), hp.dot(want))

    def test_transformation_wrt_with_parent(self):
        # With a parent attached, 'world', 'parent' and the parent object all
        # mean the same frame, while 'local' and self mean this coordinate.
        # Rotations are needed to tell them apart at all.
        parent = make_cascoords(pos=[1, 0, 0], rot=rotation_matrix(0.7, 'z'))
        child = make_cascoords(pos=[0, 1, 0], rot=rotation_matrix(-0.4, 'x'))
        parent.assoc(child, relative_coords='local')
        target = make_coords(pos=[2, 2, 2], rot=rotation_matrix(0.9, 'y'))

        hc = _homogeneous(child.worldcoords())
        ht = _homogeneous(target)
        local = np.linalg.inv(hc).dot(ht)
        world = ht.dot(np.linalg.inv(hc))

        for wrt_frame in ('local', child):
            testing.assert_almost_equal(
                _homogeneous(child.transformation(target, wrt_frame)), local,
                err_msg='wrt={}'.format(wrt_frame))
        for wrt_frame in ('world', 'parent', parent):
            testing.assert_almost_equal(
                _homogeneous(child.transformation(target, wrt_frame)), world,
                err_msg='wrt={}'.format(wrt_frame))
        # The two really are different frames here.
        self.assertFalse(np.allclose(local, world))

    def test_assoc_rejects_self_and_stolen_children(self):
        parent = make_cascoords(pos=[1, 0, 0])
        with self.assertRaises(ValueError):
            parent.assoc(parent)

        other = make_cascoords(pos=[0, 1, 0])
        child = make_cascoords(pos=[0, 0, 1])
        parent.assoc(child)
        # Re-parenting needs force; otherwise the existing link is protected.
        with self.assertRaises(RuntimeError):
            other.assoc(child)
        self.assertIs(child.parent, parent)

        world_before = _homogeneous(child.worldcoords())
        other.assoc(child, force=True)
        self.assertIs(child.parent, other)
        self.assertIn(child, other.descendants)
        # force still defaults to relative_coords='world', so it stays put.
        testing.assert_almost_equal(
            _homogeneous(child.worldcoords()), world_before)

    def test_assoc_deprecated_c_argument(self):
        parent = make_cascoords(pos=[2, 0, 0])
        child = make_cascoords(pos=[0, 2, 0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            parent.assoc(child, c='world')
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught))
        # and it means the same as relative_coords='world'
        testing.assert_almost_equal(
            child.worldcoords().translation, [0, 2, 0])

    def test_assoc_and_dissoc_preserve_world_pose(self):
        for a, b in _coords_pairs():
            hp, hc = _homogeneous(a), _homogeneous(b)

            # relative_coords='local': the child's pose is read as parent-local
            parent = make_cascoords(pos=a.translation, rot=a.rotation)
            child = make_cascoords(pos=b.translation, rot=b.rotation)
            parent.assoc(child, relative_coords='local')
            testing.assert_almost_equal(
                _homogeneous(child.worldcoords()), hp.dot(hc))

            # relative_coords='world': the child does not move
            parent = make_cascoords(pos=a.translation, rot=a.rotation)
            child = make_cascoords(pos=b.translation, rot=b.rotation)
            parent.assoc(child, relative_coords='world')
            testing.assert_almost_equal(_homogeneous(child.worldcoords()), hc)
            # ... and its stored local pose is parent^-1 * child
            testing.assert_almost_equal(
                _homogeneous(make_coords(pos=child.translation,
                                         rot=child.rotation)),
                np.linalg.inv(hp).dot(hc))
            self.assertIn(child, parent.descendants)
            self.assertIs(child.parent, parent)

            # dissoc leaves the child where it stands in the world
            parent.dissoc(child)
            testing.assert_almost_equal(_homogeneous(child.worldcoords()), hc)
            self.assertNotIn(child, parent.descendants)
            self.assertIsNone(child.parent)

    def test_child_follows_parent(self):
        for a, b in _coords_pairs():
            hp, hc = _homogeneous(a), _homogeneous(b)
            parent = make_cascoords(pos=a.translation, rot=a.rotation)
            child = make_cascoords(pos=b.translation, rot=b.rotation)
            parent.assoc(child, relative_coords='local')

            theta, axis = 1.1, np.array([0.3, 0.4, -0.5])
            axis = axis / np.linalg.norm(axis)
            hr = np.eye(4)
            hr[:3, :3] = rotation_matrix(theta, axis)
            parent.rotate(theta, axis, 'local')

            testing.assert_almost_equal(
                _homogeneous(child.worldcoords()), hp.dot(hr).dot(hc))
            # The parent moving must not touch the child's own pose.
            testing.assert_almost_equal(
                _homogeneous(make_coords(pos=child.translation,
                                         rot=child.rotation)), hc)
