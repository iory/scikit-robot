import os
import tempfile
import unittest

import numpy as np
import trimesh

from skrobot.coordinates.math import rpy2matrix
from skrobot.coordinates.math import skew_symmetric_matrix
from skrobot.utils.inertia import combine_inertials
from skrobot.utils.inertia import link_inertial_from_mesh
from skrobot.utils.inertia import mesh_mass_properties
from skrobot.utils.inertia import parallel_axis
from skrobot.utils.inertia import rescale_inertial_to_mass
from skrobot.utils.inertia import transform_inertial
from skrobot.utils.inertia import validate_inertia


def _box_tensor(mass, extents):
    """Analytic inertia of a solid box about its centre of mass."""
    a, b, c = extents
    return np.diag([mass * (b ** 2 + c ** 2) / 12.0,
                    mass * (a ** 2 + c ** 2) / 12.0,
                    mass * (a ** 2 + b ** 2) / 12.0])


def _box_mesh_file(directory, extents=(0.2, 0.1, 0.05)):
    mesh = trimesh.creation.box(extents=extents)
    path = os.path.join(directory, 'box.stl')
    mesh.export(path)
    return path, mesh


class TestMeshMassProperties(unittest.TestCase):

    def test_watertight_box_matches_analytic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = _box_mesh_file(tmp, extents=(0.2, 0.1, 0.05))
            props = mesh_mass_properties(path, density=1000.0)
            self.assertIsNotNone(props)
            mass, com, inertia, method = props
            self.assertEqual(method, 'mesh')
            volume = 0.2 * 0.1 * 0.05
            self.assertAlmostEqual(mass, 1000.0 * volume, places=6)
            np.testing.assert_allclose(com, np.zeros(3), atol=1e-9)
            # analytic solid box: I_xx = m (b^2 + c^2) / 12 etc.
            m = 1000.0 * volume
            expected = np.diag([
                m * (0.1 ** 2 + 0.05 ** 2) / 12.0,
                m * (0.2 ** 2 + 0.05 ** 2) / 12.0,
                m * (0.2 ** 2 + 0.1 ** 2) / 12.0,
            ])
            np.testing.assert_allclose(inertia, expected, rtol=1e-6,
                                       atol=1e-12)

    def test_non_watertight_falls_back_to_hull(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
            # drop one face -> not watertight
            mesh = trimesh.Trimesh(vertices=mesh.vertices,
                                   faces=mesh.faces[:-1],
                                   process=False)
            self.assertFalse(mesh.is_watertight)
            path = os.path.join(tmp, 'open_box.stl')
            mesh.export(path)
            props = mesh_mass_properties(path, density=1000.0)
            self.assertIsNotNone(props)
            self.assertEqual(props[3], 'hull')

    def test_missing_file_returns_none(self):
        self.assertIsNone(mesh_mass_properties('/nonexistent/mesh.stl'))


class TestTransformInertial(unittest.TestCase):

    def test_translation_moves_com_only(self):
        inertia = np.diag([1.0, 2.0, 3.0])
        info = transform_inertial(2.0, [0.0, 0.0, 0.0], inertia,
                                  [0.5, -0.25, 1.0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(info['com'], [0.5, -0.25, 1.0])
        np.testing.assert_allclose(info['inertia'],
                                   (1.0, 0.0, 0.0, 2.0, 0.0, 3.0),
                                   atol=1e-12)

    def test_rotation_rotates_tensor(self):
        # 90 deg about Z swaps the x and y principal moments
        inertia = np.diag([1.0, 2.0, 3.0])
        info = transform_inertial(1.0, [0.0, 0.0, 0.0], inertia,
                                  [0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2])
        ixx, ixy, ixz, iyy, iyz, izz = info['inertia']
        self.assertAlmostEqual(ixx, 2.0)
        self.assertAlmostEqual(iyy, 1.0)
        self.assertAlmostEqual(izz, 3.0)

    def test_accepts_6_components(self):
        info = transform_inertial(1.0, [0, 0, 0],
                                  (1.0, 0.0, 0.0, 2.0, 0.0, 3.0),
                                  [0, 0, 0], [0, 0, 0])
        np.testing.assert_allclose(info['inertia'],
                                   (1.0, 0.0, 0.0, 2.0, 0.0, 3.0))

    def test_invalid_returns_none(self):
        self.assertIsNone(transform_inertial(None, [0, 0, 0],
                                             np.eye(3), [0, 0, 0], [0, 0, 0]))
        self.assertIsNone(transform_inertial(-1.0, [0, 0, 0],
                                             np.eye(3), [0, 0, 0], [0, 0, 0]))


class TestLinkInertialFromMesh(unittest.TestCase):

    def test_visual_origin_is_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = _box_mesh_file(tmp)
            xyz = [0.1, 0.2, 0.3]
            rpy = [0.3, -0.2, 0.5]
            info = link_inertial_from_mesh(path, xyz, rpy, density=500.0)
            self.assertEqual(info['method'], 'mesh')
            # box com is at the mesh origin -> link-frame com equals xyz
            np.testing.assert_allclose(info['com'], xyz, atol=1e-9)
            # tensor must stay physically valid under the rotation
            self.assertEqual(validate_inertia(info['mass'],
                                              info['inertia']), [])
            # and equal the rotated diagonal tensor
            props = mesh_mass_properties(path, density=500.0)
            rot = rpy2matrix(*rpy)
            expected = rot @ props[2] @ rot.T
            got = info['inertia']
            np.testing.assert_allclose(
                [got[0], got[3], got[5]],
                [expected[0, 0], expected[1, 1], expected[2, 2]],
                rtol=1e-9)

    def test_none_path_returns_none(self):
        self.assertIsNone(link_inertial_from_mesh(None, [0, 0, 0], [0, 0, 0]))


class TestRescaleAndValidate(unittest.TestCase):

    def test_rescale(self):
        info = {'mass': 2.0, 'com': [1.0, 2.0, 3.0],
                'inertia': (1.0, 0.0, 0.0, 2.0, 0.0, 3.0), 'method': 'mesh'}
        out = rescale_inertial_to_mass(info, 4.0)
        self.assertEqual(out['mass'], 4.0)
        self.assertEqual(out['com'], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(out['inertia'],
                                   (2.0, 0.0, 0.0, 4.0, 0.0, 6.0))
        self.assertEqual(out['method'], 'mesh->mass')
        # invalid target leaves input unchanged
        self.assertIs(rescale_inertial_to_mass(info, -1.0), info)

    def test_validate(self):
        self.assertEqual(
            validate_inertia(1.0, (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)), [])
        self.assertTrue(
            validate_inertia(-1.0, (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)))
        # triangle inequality violation: I1 + I2 < I3
        problems = validate_inertia(1.0, (1.0, 0.0, 0.0, 1.0, 0.0, 3.0))
        self.assertTrue(any('triangle' in p for p in problems))
        # not positive definite
        problems = validate_inertia(1.0, (-1.0, 0.0, 0.0, 1.0, 0.0, 1.0))
        self.assertTrue(any('positive definite' in p for p in problems))


class TestParallelAxis(unittest.TestCase):

    def test_point_mass_offset(self):
        # a 2 kg body 1 m along x gains m * d^2 about y and z, nothing about x
        shifted = parallel_axis(2.0, np.zeros((3, 3)), [1, 0, 0], [0, 0, 0])
        np.testing.assert_allclose(shifted, np.diag([0.0, 2.0, 2.0]))

    def test_zero_offset_is_identity(self):
        tensor = np.diag([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            parallel_axis(5.0, tensor, [1, 2, 3], [1, 2, 3]), tensor)

    def test_box_about_corner_matches_textbook(self):
        # a solid box about one corner is m*(b^2+c^2)/3 (vs /12 about the com)
        mass, extents = 3.0, (0.2, 0.1, 0.05)
        a, b, c = extents
        corner = parallel_axis(mass, _box_tensor(mass, extents),
                               [0.0, 0.0, 0.0],
                               [a / 2.0, b / 2.0, c / 2.0])
        np.testing.assert_allclose(
            np.diag(corner),
            [mass * (b ** 2 + c ** 2) / 3.0,
             mass * (a ** 2 + c ** 2) / 3.0,
             mass * (a ** 2 + b ** 2) / 3.0], atol=1e-12)
        # the products of inertia about a corner are -m*x*y etc.
        self.assertAlmostEqual(corner[0, 1], -mass * (a / 2.0) * (b / 2.0))

    def test_accepts_6_components(self):
        components = (1.0, 0.1, 0.2, 2.0, 0.3, 3.0)
        tensor = np.array([[1.0, 0.1, 0.2],
                           [0.1, 2.0, 0.3],
                           [0.2, 0.3, 3.0]])
        np.testing.assert_allclose(
            parallel_axis(1.5, components, [0, 1, 0], [0, 0, 0]),
            parallel_axis(1.5, tensor, [0, 1, 0], [0, 0, 0]))

    def test_matches_skew_symmetric_form(self):
        # the same theorem written as -m [r]x [r]x, as used by
        # RobotModel.update_mass_properties
        rng = np.random.RandomState(0)
        for _ in range(20):
            mass = float(rng.uniform(0.01, 10.0))
            offset = rng.uniform(-2.0, 2.0, 3)
            tensor = rng.uniform(-1.0, 1.0, (3, 3))
            tensor = tensor + tensor.T
            cross = skew_symmetric_matrix(offset)
            np.testing.assert_allclose(
                parallel_axis(mass, tensor, offset, np.zeros(3)),
                tensor - mass * cross.dot(cross), atol=1e-12)


class TestCombineInertials(unittest.TestCase):

    def test_two_point_masses(self):
        zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        left = {'mass': 1.0, 'com': [-1, 0, 0], 'inertia': zero}
        right = {'mass': 1.0, 'com': [1, 0, 0], 'inertia': zero}
        out = combine_inertials(left, right)
        self.assertEqual(out['mass'], 2.0)
        np.testing.assert_allclose(out['com'], [0.0, 0.0, 0.0], atol=1e-12)
        # each mass sits 1 m off the combined centre, on the x axis
        np.testing.assert_allclose(out['inertia'], (0.0, 0.0, 0.0,
                                                    2.0, 0.0, 2.0),
                                   atol=1e-12)

    def test_halves_recombine_into_the_whole_box(self):
        """Splitting a solid box in two and lumping it back must reproduce the
        box's own tensor -- the physical invariant this function exists for."""
        extents = (0.2, 0.1, 0.05)
        mass = 4.0
        half = {'mass': mass / 2.0,
                'inertia': _box_tensor(mass / 2.0,
                                       (extents[0] / 2.0,) + extents[1:])}
        out = combine_inertials(
            dict(half, com=[-extents[0] / 4.0, 0, 0]),
            dict(half, com=[extents[0] / 4.0, 0, 0]))
        self.assertAlmostEqual(out['mass'], mass)
        np.testing.assert_allclose(out['com'], [0.0, 0.0, 0.0], atol=1e-12)
        whole = _box_tensor(mass, extents)
        np.testing.assert_allclose(
            out['inertia'],
            (whole[0, 0], 0.0, 0.0, whole[1, 1], 0.0, whole[2, 2]),
            atol=1e-12)

    def test_order_does_not_matter(self):
        a = {'mass': 1.5, 'com': [0.1, 0.2, 0.3],
             'inertia': (1.0, 0.0, 0.0, 2.0, 0.0, 3.0), 'method': 'a'}
        b = {'mass': 0.5, 'com': [-0.4, 0.0, 0.2],
             'inertia': (0.5, 0.1, 0.0, 0.5, 0.0, 0.5), 'method': 'b'}
        forward = combine_inertials(a, b)
        backward = combine_inertials(b, a)
        np.testing.assert_allclose(forward['inertia'], backward['inertia'],
                                   atol=1e-12)
        np.testing.assert_allclose(forward['com'], backward['com'], atol=1e-12)

    def test_skips_unusable_and_reports_method(self):
        good = {'mass': 2.0, 'com': [0, 0, 0],
                'inertia': (1.0, 0.0, 0.0, 1.0, 0.0, 1.0), 'method': 'mesh'}
        massless = {'mass': 0.0, 'com': [1, 1, 1],
                    'inertia': (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)}
        out = combine_inertials(good, None, massless, dict(good, method='sw'))
        self.assertEqual(out['mass'], 4.0)
        self.assertEqual(out['method'], 'mesh+sw')

    def test_returns_none_when_nothing_usable(self):
        self.assertIsNone(combine_inertials())
        self.assertIsNone(combine_inertials(None, {}))
        self.assertIsNone(combine_inertials(
            {'mass': -1.0, 'com': [0, 0, 0],
             'inertia': (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)}))

    def test_composes_with_transform_inertial(self):
        """The documented workflow: move each body into a common frame with
        transform_inertial, then lump them."""
        body = {'mass': 1.0, 'com': [0.1, 0.0, 0.0],
                'inertia': (0.01, 0.0, 0.0, 0.02, 0.0, 0.03)}
        moved = transform_inertial(body['mass'], body['com'], body['inertia'],
                                   [0.0, 0.0, 0.5], [0.0, 0.0, np.pi / 2])
        out = combine_inertials(body, moved)
        self.assertAlmostEqual(out['mass'], 2.0)
        # com of the rotated+translated copy is Rz(90) @ [0.1,0,0] + [0,0,0.5]
        np.testing.assert_allclose(out['com'],
                                   [0.05, 0.05, 0.25], atol=1e-12)
        self.assertEqual(validate_inertia(out['mass'], out['inertia']), [])


if __name__ == '__main__':
    unittest.main()
