import os
import tempfile
import unittest

import numpy as np
import trimesh

from skrobot.coordinates.math import rpy2matrix
from skrobot.utils.inertia import link_inertial_from_mesh
from skrobot.utils.inertia import mesh_mass_properties
from skrobot.utils.inertia import rescale_inertial_to_mass
from skrobot.utils.inertia import transform_inertial
from skrobot.utils.inertia import validate_inertia


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


if __name__ == '__main__':
    unittest.main()
