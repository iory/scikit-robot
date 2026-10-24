import unittest

import numpy as np
import trimesh

import skrobot
from skrobot.collision.robot_collision import RobotCollisionChecker
from skrobot.planner.swept_sphere import compute_fitted_spheres
from skrobot.planner.swept_sphere import compute_swept_sphere


class TestSweptSphere(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        robot_model = skrobot.models.PR2()
        cls.mesh1 = robot_model.r_gripper_palm_link._visual_mesh[0]

        objfile_path = skrobot.data.bunny_objpath()
        bunnymesh = trimesh.load_mesh(objfile_path)
        cls.mesh2 = bunnymesh

    def test_copmute_swept_sphere(self):

        def approximation_accuracy_test(mesh, tol):
            center_pts, radius = compute_swept_sphere(mesh, tol=tol)

            def swept_sphere_sdf(pts):
                dists_list = []
                for c in center_pts:
                    dists = np.sqrt(np.sum((pts - c[None, :])**2, axis=1))
                    dists_list.append(dists)
                dists_arr = np.array(dists_list)
                # union sdf of all spheres
                signed_dists = np.min(dists_arr, axis=0) - radius
                return signed_dists

            jut_arr = swept_sphere_sdf(mesh.vertices)
            max_jut = np.max(jut_arr)
            self.assertLess(max_jut / radius, tol)

        tol = 1e-3
        approximation_accuracy_test(self.mesh1, tol)
        approximation_accuracy_test(self.mesh2, tol)


def _surface_samples(mesh, n=5000):
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    return np.concatenate([pts, mesh.vertices])


def _check_fit(test, mesh, tol):
    """Every surface point is in a sphere; no sphere reaches more than
    tol + 2 res out of the convex hull."""
    centers, radii = compute_fitted_spheres(mesh, tol=tol)
    res = max(tol / 2, float(np.max(mesh.extents)) / 64)
    pts = _surface_samples(mesh)
    d = np.linalg.norm(pts[:, None] - centers[None], axis=2) - radii[None]
    test.assertLessEqual(d.min(axis=1).max(), 1e-9)
    # a sphere's farthest reach out of the hull: its radius less how deep
    # its centre is inside (the hull's planes, the closest one)
    hull = mesh.convex_hull
    depth = -(centers @ hull.face_normals.T
              - np.einsum('ij,ij->i', hull.face_normals, hull.triangles[:, 0]))
    out = radii - depth.min(axis=1)
    test.assertLessEqual(out.max(), tol + 2 * res + 1e-9)
    return centers, radii


class TestFittedSpheres(unittest.TestCase):

    def test_box(self):
        box = trimesh.creation.box(extents=[0.3, 0.2, 0.1])
        centers, radii = _check_fit(self, box, 0.01)
        # spheres of one size every res along the faces would take hundreds;
        # this fit fewer, with big ones inside
        self.assertLess(len(centers), 200)
        self.assertGreater(radii.max(), 2 * radii.min())

    def test_robot_link(self):
        mesh = skrobot.models.PR2().r_gripper_palm_link._visual_mesh[0]
        _check_fit(self, mesh, 0.01)

    def test_solid_mesh(self):
        box = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        centers, _ = compute_fitted_spheres(box, tol=0.01, solid='mesh')
        self.assertGreater(len(centers), 0)
        with self.assertRaises(ValueError):
            compute_fitted_spheres(box, solid='shell')

    def test_add_link(self):
        robot = skrobot.models.PR2()
        link = robot.r_gripper_palm_link
        checker = RobotCollisionChecker(robot)
        checker.add_link(link, geometry_type='fitted_spheres', fit_tol=0.01)
        geoms = [g.geometry for g in checker._link_geometries]
        _, radii = compute_fitted_spheres(link.collision_mesh, tol=0.01)
        self.assertEqual(len(geoms), len(radii))
        np.testing.assert_allclose(sorted(float(g.radius) for g in geoms),
                                   sorted(radii))
