import unittest

import numpy as np
import trimesh

from skrobot.coordinates.math import rpy_matrix
from skrobot.utils.primitive_fitting import fit_primitive_to_mesh
from skrobot.utils.primitive_fitting import primitive_params_to_origin


def _rotated_box(extents, roll, pitch, yaw, center):
    """Create a box mesh rotated by (roll, pitch, yaw) and translated."""
    mesh = trimesh.creation.box(extents=extents)
    rotation = rpy_matrix(yaw, pitch, roll)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    mesh.apply_transform(transform)
    return mesh, rotation


class TestFitBox(unittest.TestCase):

    def test_rotated_box_auto_selects_oriented_fit(self):
        # A clearly rotated box: the oriented candidate is much tighter than
        # the axis-aligned one, so auto-selection must return it.
        true_extents = np.array([0.30, 0.12, 0.06])
        center = np.array([0.5, -0.2, 1.0])
        mesh, _ = _rotated_box(
            true_extents, roll=0.3, pitch=-0.4, yaw=0.5, center=center)

        params = fit_primitive_to_mesh(mesh, primitive_type='box')

        self.assertEqual(params['type'], 'box')
        # Extents match the true box dimensions (OBB order is arbitrary).
        np.testing.assert_allclose(
            np.sort(params['extents']), np.sort(true_extents), atol=1e-6)
        np.testing.assert_allclose(params['center'], center, atol=1e-6)
        # A proper, non-identity rotation was recovered.
        self.assertFalse(np.allclose(params['rotation'], np.eye(3)))
        self.assertAlmostEqual(
            np.linalg.det(params['rotation']), 1.0, places=6)
        np.testing.assert_allclose(
            params['rotation'] @ params['rotation'].T, np.eye(3), atol=1e-6)

    def test_axis_aligned_box_recovers_extents(self):
        true_extents = np.array([0.30, 0.12, 0.06])
        center = np.array([0.5, -0.2, 1.0])
        mesh = trimesh.creation.box(extents=true_extents)
        mesh.apply_translation(center)

        params = fit_primitive_to_mesh(mesh, primitive_type='box')

        np.testing.assert_allclose(
            np.sort(params['extents']), np.sort(true_extents), atol=1e-6)
        np.testing.assert_allclose(params['center'], center, atol=1e-6)

    def test_oriented_flag_forces_orientation(self):
        # Same rotated box, forced axis-aligned vs forced oriented.
        true_extents = np.array([0.30, 0.12, 0.06])
        center = np.array([0.5, -0.2, 1.0])
        mesh, _ = _rotated_box(
            true_extents, roll=0.3, pitch=-0.4, yaw=0.5, center=center)

        aabb = fit_primitive_to_mesh(
            mesh, primitive_type='box', oriented=False)
        obb = fit_primitive_to_mesh(
            mesh, primitive_type='box', oriented=True)

        # oriented=False -> axis-aligned (identity rotation, loose extents).
        np.testing.assert_allclose(aabb['rotation'], np.eye(3), atol=1e-9)
        # oriented=True -> real OBB recovering the true extents / rotation.
        self.assertFalse(np.allclose(obb['rotation'], np.eye(3)))
        np.testing.assert_allclose(
            np.sort(obb['extents']), np.sort(true_extents), atol=1e-6)


class TestFitCylinder(unittest.TestCase):

    def test_tilted_cylinder_auto_recovers_axis(self):
        radius = 0.1
        height = 0.5
        axis = np.array([0.3, 0.4, np.sqrt(1.0 - 0.25)])
        axis /= np.linalg.norm(axis)

        mesh = trimesh.creation.cylinder(radius=radius, height=height)
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, axis)
        s = np.linalg.norm(v)
        c = np.dot(z, axis)
        skew = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        rotation = np.eye(3) + skew + skew @ skew * ((1 - c) / (s ** 2))
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.array([0.2, 0.3, 0.4])
        mesh.apply_transform(transform)

        params = fit_primitive_to_mesh(mesh, primitive_type='cylinder')

        self.assertEqual(params['type'], 'cylinder')
        self.assertAlmostEqual(params['radius'], radius, places=2)
        self.assertAlmostEqual(params['height'], height, places=2)
        fitted_axis = params['axis'] / np.linalg.norm(params['axis'])
        # Axis is recovered up to sign.
        self.assertAlmostEqual(abs(np.dot(fitted_axis, axis)), 1.0, places=2)


class TestAutoTypeSelection(unittest.TestCase):

    def test_boxy_mesh_selects_box(self):
        mesh = trimesh.creation.box(extents=[0.3, 0.12, 0.06])
        params = fit_primitive_to_mesh(mesh)
        self.assertEqual(params['type'], 'box')

    def test_round_mesh_selects_sphere(self):
        mesh = trimesh.creation.icosphere(radius=0.2, subdivisions=3)
        params = fit_primitive_to_mesh(mesh)
        self.assertEqual(params['type'], 'sphere')

    def test_cylindrical_mesh_selects_cylinder(self):
        mesh = trimesh.creation.cylinder(radius=0.1, height=0.6)
        params = fit_primitive_to_mesh(mesh)
        self.assertEqual(params['type'], 'cylinder')


class TestPrimitiveParamsToOrigin(unittest.TestCase):

    def test_box_roundtrips_center_and_rpy(self):
        roll, pitch, yaw = 0.2, 0.35, -0.6
        rotation = rpy_matrix(yaw, pitch, roll)
        center = np.array([1.0, 2.0, 3.0])
        params = {
            'type': 'box',
            'center': center,
            'extents': np.array([1.0, 2.0, 3.0]),
            'rotation': rotation,
        }

        xyz, rpy = primitive_params_to_origin(params)

        np.testing.assert_allclose(xyz, center)
        np.testing.assert_allclose(rpy, [roll, pitch, yaw], atol=1e-9)
        # rpy reconstructs the original rotation.
        np.testing.assert_allclose(
            rpy_matrix(rpy[2], rpy[1], rpy[0]), rotation, atol=1e-9)

    def test_oriented_box_origin_reconstructs_rotation(self):
        true_extents = np.array([0.30, 0.12, 0.06])
        center = np.array([0.5, -0.2, 1.0])
        mesh, _ = _rotated_box(
            true_extents, roll=0.3, pitch=-0.4, yaw=0.5, center=center)
        params = fit_primitive_to_mesh(mesh, primitive_type='box')

        xyz, rpy = primitive_params_to_origin(params)

        np.testing.assert_allclose(xyz, params['center'])
        np.testing.assert_allclose(
            rpy_matrix(rpy[2], rpy[1], rpy[0]), params['rotation'], atol=1e-6)

    def test_sphere_has_zero_rotation(self):
        center = np.array([1.0, 2.0, 3.0])
        params = {'type': 'sphere', 'center': center, 'radius': 0.5}

        xyz, rpy = primitive_params_to_origin(params)

        np.testing.assert_allclose(xyz, center)
        np.testing.assert_allclose(rpy, np.zeros(3))

    def test_cylinder_axis_maps_to_z(self):
        axis = np.array([0.0, 1.0, 0.0])
        center = np.array([0.1, 0.2, 0.3])
        params = {
            'type': 'cylinder',
            'center': center,
            'radius': 0.1,
            'height': 0.5,
            'axis': axis,
        }

        xyz, rpy = primitive_params_to_origin(params)

        np.testing.assert_allclose(xyz, center)
        rotation = rpy_matrix(rpy[2], rpy[1], rpy[0])
        # The primitive local Z-axis is aligned with the fitted axis.
        np.testing.assert_allclose(rotation @ [0, 0, 1], axis, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
