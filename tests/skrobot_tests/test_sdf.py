import numpy as np
from numpy import testing
import os
import shutil
import trimesh
import unittest

import skrobot
from skrobot.sdf import BoxSDF
from skrobot.sdf import CylinderSDF
from skrobot.sdf import GridSDF
from skrobot.sdf import SphereSDF
from skrobot.sdf import UnionSDF


class TestSDF(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        # clear cache
        home_dir = os.path.expanduser("~")
        sdf_cache_dir = os.path.join(home_dir, '.skrobot', 'sdf')
        if os.path.exists(sdf_cache_dir):
            shutil.rmtree(sdf_cache_dir)

        # prepare gridsdf
        objfile_path = skrobot.data.bunny_objpath()
        bunnymesh = trimesh.load_mesh(objfile_path)
        gridsdf = GridSDF.from_objfile(objfile_path, dim_grid=50)
        cls.gridsdf = gridsdf
        cls.bunnymesh = bunnymesh

        # prepare boxsdf
        box_withds = np.array([0.05, 0.1, 0.05])
        boxsdf = BoxSDF([0, 0, 0], box_withds)
        boxtrans = np.array([0.0, 0.1, 0.0])
        boxsdf.coords.translate(boxtrans)

        cls.box_withds = box_withds
        cls.boxsdf = boxsdf
        cls.boxtrans = boxtrans
        cls.points_box_edge_sdf = \
            np.array([-0.5 * box_withds, 0.5 * box_withds])

        # prepare SphereSDF
        radius = 1.0
        height = 1.0
        cls.radius = radius
        cls.height = height
        cls.sphere_sdf = SphereSDF([0, 0, 0], radius=radius)
        cls.cylinder_sdf = CylinderSDF([0, 0, 0], radius=radius, height=height)

        # preapare UnionSDF
        unionsdf = UnionSDF(sdf_list=[boxsdf, gridsdf])
        cls.unionsdf = unionsdf

    def test_box__signed_distance(self):
        sdf = self.boxsdf
        X_origin = np.zeros((1, 3))
        self.assertEqual(sdf._signed_distance(
            X_origin), -min(self.box_withds) * 0.5)
        testing.assert_array_equal(
            sdf._signed_distance(self.points_box_edge_sdf), [0, 0])

    def test_box__surface_points(self):
        sdf = self.boxsdf
        ray_tips, sd_vals = sdf._surface_points()
        testing.assert_array_equal(sdf._signed_distance(ray_tips), sd_vals)
        self.assertTrue(np.all(np.abs(sd_vals) < sdf._surface_threshold))

    def test_sphere__signed_distance(self):
        sdf, radius = self.sphere_sdf, self.radius
        vecs = np.random.randn(100, 3)
        norm = np.sqrt(np.sum(vecs**2, axis=1))
        vecs_unit = radius * vecs / norm[:, None]
        sd_vals = sdf(vecs_unit)
        testing.assert_almost_equal(sd_vals, np.zeros(100))
        self.assertEqual(sdf(np.zeros((1, 3))).item(), -radius)

    def test_sphere__surface_points(self):
        sdf = self.sphere_sdf
        ray_tips, sd_vals = sdf._surface_points()
        testing.assert_array_equal(sdf._signed_distance(ray_tips), sd_vals)
        self.assertTrue(np.all(np.abs(sd_vals) < sdf._surface_threshold))

    def test_cylinder__signed_distance(self):
        sdf, radius, height = self.cylinder_sdf, self.radius, self.height
        half_height = height * 0.5
        pts_on_surface = np.array([
            [0, 0, half_height],
            [radius, 0, half_height],
            [radius, 0, half_height],
            [radius, 0, 0],
            [0, 0, -half_height],
            [radius, 0, -half_height],
            [radius, 0, -half_height]
        ])
        testing.assert_almost_equal(sdf(pts_on_surface),
                                    np.zeros(len(pts_on_surface)))
        self.assertEqual(sdf(np.zeros((1, 3))).item(),
                         -min(radius, half_height))

    def test_cylinder__surface_points(self):
        sdf = self.cylinder_sdf
        ray_tips, sd_vals = sdf._surface_points()
        testing.assert_array_equal(sdf._signed_distance(ray_tips), sd_vals)
        self.assertTrue(np.all(np.abs(sd_vals) < sdf._surface_threshold))

    def test__transform_pts_obj_to_sdf_and_sdf_to_obj(self):
        sdf, trans = self.boxsdf, self.boxtrans
        points_obj = np.random.randn(100, 3)
        points_sdf = sdf._transform_pts_obj_to_sdf(points_obj)

        # test transform_pts_obj_to_sdf
        points_sdf_should_be = points_obj - \
            np.repeat(trans.reshape((1, -1)), 100, axis=0)
        testing.assert_array_almost_equal(points_sdf, points_sdf_should_be)

        # test transform_pts_sdf_to_obj
        points_obj_recreated = sdf._transform_pts_sdf_to_obj(points_sdf)
        testing.assert_array_almost_equal(points_obj_recreated, points_obj)

    def test___call__(self):
        sdf, trans = self.boxsdf, self.boxtrans
        points_box_edge_obj = np.array(
            [x + trans for x in self.points_box_edge_sdf])
        testing.assert_array_almost_equal(
            sdf(points_box_edge_obj), [0, 0])

    def test_surface_points(self):
        sdf = self.boxsdf
        surface_points_obj, _ = sdf.surface_points(n_sample=20)
        sdf_vals = sdf(surface_points_obj)
        self.assertTrue(np.all(np.abs(sdf_vals) < sdf._surface_threshold))

    def test_on_surface(self):
        sdf = self.boxsdf
        points_box_edge_obj = sdf._transform_pts_sdf_to_obj(
            self.points_box_edge_sdf)
        logicals_positive, _ = sdf.on_surface(points_box_edge_obj)
        self.assertTrue(np.all(logicals_positive))

        points_origin = np.zeros((1, 3))
        logicals_negative, _ = sdf.on_surface(points_origin)
        self.assertTrue(np.all(~logicals_negative))

    def test_gridsdf_is_out_of_bounds(self):
        sdf, mesh = self.gridsdf, self.bunnymesh
        vertices_obj = mesh.vertices
        b_min = np.min(vertices_obj, axis=0)
        b_max = np.max(vertices_obj, axis=0)
        center = 0.5 * (b_min + b_max)
        width = b_max - b_min
        points_outer_bbox = np.array([
            center + width,
            center - width
        ])
        # this condition maybe depends on the padding when creating sdf
        self.assertTrue(np.all(sdf.is_out_of_bounds(points_outer_bbox)))
        self.assertTrue(np.all(~sdf.is_out_of_bounds(vertices_obj)))

    def test_gridsdf__signed_distance(self):
        sdf, mesh = self.gridsdf, self.bunnymesh
        vertices_obj = mesh.vertices
        vertices_sdf = sdf._transform_pts_obj_to_sdf(vertices_obj)
        sd_vals = sdf._signed_distance(vertices_sdf)
        # all vertices of the mesh must be on the surface
        assert np.all(np.abs(sd_vals) < sdf._surface_threshold)

        # sd of points outside of bounds must be np.inf
        point_outofbound = (sdf._dims + 1).reshape(1, 3)
        sd_vals = sdf._signed_distance(point_outofbound)
        assert np.all(np.isinf(sd_vals))

    def test_gridsdf_surface_points(self):
        sdf, _ = self.gridsdf, self.bunnymesh
        surf_points_obj, _ = sdf.surface_points()
        logicals, _ = sdf.on_surface(surf_points_obj)
        assert np.all(logicals)

    def test_unionsdf_assert_use_abs_false(self):
        b1 = BoxSDF([1, 1, 1], [0, 0, 0], use_abs=True)
        b2 = BoxSDF([1, 1, 1], [0, 0, 0], use_abs=False)
        with self.assertRaises(AssertionError):
            UnionSDF(sdf_list=[b1, b2])

    def test_unionsdf___call__(self):
        sdf = self.unionsdf
        pts_on_surface = np.array([
            [-0.07196818, 0.16532058, -0.04285806],
            [0.02802324, 0.11360088, -0.00837826],
            [-0.05472828, 0.03257335, 0.00886164],
            [0.0077233, 0.15, -0.01742908],
            [0.02802324, 0.11360088, -0.00837826],
            [-0.07714015, 0.15152866, 0.0329975]
        ])
        sd_vals = sdf(pts_on_surface)
        self.assertTrue(np.all(abs(sd_vals) < sdf._surface_threshold))

    def test_unionsdf_surface_points(self):
        sdf = self.unionsdf
        sdf.surface_points()
        pts, sd_vals = sdf.surface_points()
        self.assertTrue(np.all(np.abs(sd_vals) < sdf._surface_threshold))

        sub_sdf1, sub_sdf2 = self.unionsdf.sdf_list
        on_surface1 = sub_sdf1(pts) < sub_sdf1._surface_threshold
        on_surface2 = sub_sdf2(pts) < sub_sdf2._surface_threshold
        cond_or = np.logical_or(on_surface1, on_surface2)
        self.assertTrue(np.all(cond_or))  # at least on either of the surface

        cond_and = (sum(on_surface1) > 0) and (sum(on_surface2) > 0)
        self.assertTrue(cond_and)  # each surface has at least a single points

    def test_union_sdf_from_robot_model(self):
        r2d2 = skrobot.models.urdf.RobotModelFromURDF(
            urdf_file=skrobot.data.r2d2_urdfpath())
        r2d2_union_sdf = UnionSDF.from_robot_model(r2d2)
        for link in r2d2.link_list:
            coll_mesh = link._collision_mesh
            vertices = link.transform_vector(coll_mesh.vertices)
            sd_vals = r2d2_union_sdf(vertices)
            self.assertTrue(np.all(sd_vals < 1e-3))

        fetch = skrobot.models.Fetch()
        fetch.reset_manip_pose()
        # this a takes
        fetch_union_sdf = UnionSDF.from_robot_model(fetch)

        # check if the vertices of the links have almost 0 sd vals.
        gripper_link = fetch.gripper_link
        coll_mesh = gripper_link._collision_mesh
        vertices = gripper_link.transform_vector(coll_mesh.vertices)
        sd_vals = fetch_union_sdf(vertices)
        self.assertTrue(np.all(sd_vals < 1e-3))

        finger_link = fetch.r_gripper_finger_link
        coll_mesh = finger_link._collision_mesh
        vertices = finger_link.transform_vector(coll_mesh.vertices)
        sd_vals = fetch_union_sdf(vertices)
        self.assertTrue(np.all(sd_vals < 1e-3))
