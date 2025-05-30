import os
import shutil
import tempfile
import unittest

import numpy as np
from numpy import testing
import trimesh

import skrobot
from skrobot.data import bunny_objpath
from skrobot.data import get_cache_dir
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.sdf import BoxSDF
from skrobot.sdf import CylinderSDF
from skrobot.sdf import GridSDF
from skrobot.sdf import SphereSDF
from skrobot.sdf import UnionSDF


class TestSDF(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        # clear cache
        sdf_cache_dir = os.path.join(get_cache_dir(), 'sdf')
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
        boxsdf = BoxSDF(box_withds)
        boxtrans = np.array([0.0, 0.1, 0.0])
        boxsdf.translate(boxtrans)

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
        cls.sphere_sdf = SphereSDF(radius=radius)
        cls.cylinder_sdf = CylinderSDF(radius=radius, height=height)

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

    def test__transform_pts_world_to_sdf_and_sdf_to_world(self):
        sdf, trans = self.boxsdf, self.boxtrans
        points_world = np.random.randn(100, 3)
        points_sdf = sdf._transform_pts_world_to_sdf(points_world)

        # test transform_pts_world_to_sdf
        points_sdf_should_be = points_world - \
            np.repeat(trans.reshape((1, -1)), 100, axis=0)
        testing.assert_array_almost_equal(points_sdf, points_sdf_should_be)

        # test transform_pts_sdf_to_world
        points_world_recreated = sdf._transform_pts_sdf_to_world(points_sdf)
        testing.assert_array_almost_equal(points_world_recreated, points_world)

    def test___call__(self):
        sdf, trans = self.boxsdf, self.boxtrans
        points_box_edge_world = np.array(
            [x + trans for x in self.points_box_edge_sdf])
        testing.assert_array_almost_equal(
            sdf(points_box_edge_world), [0, 0])

    def test_update(self):
        origins = np.zeros((1, 3))
        width = np.array([1, 1, 1])
        sdf = BoxSDF(width)
        testing.assert_almost_equal(sdf(origins)[0], -0.5)

        # after translation, transformations must be updated
        sdf.translate(0.5 * width)
        testing.assert_almost_equal(sdf(origins)[0], 0.0)

    def test_surface_points(self):
        sdf = self.boxsdf
        surface_points_world, _ = sdf.surface_points(n_sample=20)
        sdf_vals = sdf(surface_points_world)
        self.assertTrue(np.all(np.abs(sdf_vals) < sdf._surface_threshold))

    def test_on_surface(self):
        sdf = self.boxsdf
        points_box_edge_world = sdf._transform_pts_sdf_to_world(
            self.points_box_edge_sdf)
        logicals_positive, _ = sdf.on_surface(points_box_edge_world)
        self.assertTrue(np.all(logicals_positive))

        points_origin = np.zeros((1, 3))
        logicals_negative, _ = sdf.on_surface(points_origin)
        self.assertTrue(np.all(~logicals_negative))

    def test_gridsdf_is_out_of_bounds(self):
        sdf, mesh = self.gridsdf, self.bunnymesh
        vertices_world = mesh.vertices
        b_min = np.min(vertices_world, axis=0)
        b_max = np.max(vertices_world, axis=0)
        center = 0.5 * (b_min + b_max)
        width = b_max - b_min
        points_outer_bbox = np.array([
            center + width,
            center - width
        ])
        # this condition maybe depends on the padding when creating sdf
        self.assertTrue(np.all(sdf.is_out_of_bounds(points_outer_bbox)))
        self.assertTrue(np.all(~sdf.is_out_of_bounds(vertices_world)))

    def test_gridsdf__signed_distance(self):
        sdf, mesh = self.gridsdf, self.bunnymesh
        vertices_world = mesh.vertices
        vertices_sdf = sdf._transform_pts_world_to_sdf(vertices_world)
        sd_vals = sdf._signed_distance(vertices_sdf)
        # all vertices of the mesh must be on the surface
        assert np.all(np.abs(sd_vals) < sdf._surface_threshold)

        # sd of points outside of bounds must be np.inf
        point_outofbound = (sdf._dims + 1).reshape(1, 3)
        sd_vals = sdf._signed_distance(point_outofbound)
        assert np.all(np.isinf(sd_vals))

    def test_gridsdf_surface_points(self):
        sdf, _ = self.gridsdf, self.bunnymesh
        surf_points_world, _ = sdf.surface_points()
        logicals, _ = sdf.on_surface(surf_points_world)
        assert np.all(logicals)

    def test_unionsdf_assert_use_abs_false(self):
        b1 = BoxSDF([1, 1, 1], use_abs=True)
        b2 = BoxSDF([1, 1, 1], use_abs=False)
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
        here_full_filepath = os.path.join(os.getcwd(), __file__)
        here_full_dirpath = os.path.dirname(here_full_filepath)
        test_model_path = os.path.join(
            here_full_dirpath, 'data', 'primitives.urdf')

        primitive_robot = skrobot.models.urdf.RobotModelFromURDF(
            urdf_file=test_model_path)
        primitive_robot_sdf = UnionSDF.from_robot_model(primitive_robot)
        for link in primitive_robot.link_list:
            coll_mesh = link._collision_mesh
            vertices = link.transform_vector(coll_mesh.vertices)
            sd_vals = primitive_robot_sdf(vertices)
            self.assertTrue(np.all(sd_vals < 1e-3))

        fetch = skrobot.models.Fetch()
        fetch.reset_manip_pose()
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

    def test_sdf_from_robot_with_scale_parameter(self):
        td = tempfile.mkdtemp()
        urdf_file = """
        <robot name="myfirst">
          <link name="base_link">
            <visual>
              <geometry>
                <mesh filename="./bunny.obj" scale="10 10 10" />
              </geometry>
            </visual>
            <collision>
              <geometry>
                <mesh filename="./bunny.obj" scale="10 10 10" />
              </geometry>
            </collision>
          </link>
        </robot>
        """
        # write urdf file
        with open(os.path.join(td, 'temp.urdf'), 'w') as f:
            f.write(urdf_file)

        shutil.copy(bunny_objpath(), os.path.join(td, 'bunny.obj'))
        urdf_file = os.path.join(td, 'temp.urdf')
        dummy_robot = RobotModelFromURDF(urdf_file=urdf_file)
        robot_sdf = UnionSDF.from_robot_model(dummy_robot, dim_grid=100)

        bunny_mesh = trimesh.load_mesh(bunny_objpath())
        vertices = bunny_mesh.vertices * 10.0
        sd_vals = robot_sdf(vertices)

        # If sdf is properly computed, abs(sdf) for all vertices
        # must be small enough
        lb, lu = np.min(vertices, axis=0), np.max(vertices, axis=0)
        eps = np.min(lu - lb) * 1e-2
        self.assertTrue(np.all(np.abs(sd_vals) < eps))

    def test_trimesh2sdf(self):
        # non-primitive mesh with file_path
        mesh = self.bunnymesh
        sdf = skrobot.sdf.trimesh2sdf(mesh)
        assert isinstance(sdf, GridSDF)

        # non-primitive mesh without file_path
        sdf = skrobot.sdf.trimesh2sdf(mesh)
        assert isinstance(sdf, GridSDF)

        # primitive mesh (box)
        mesh = trimesh.creation.box([1, 1, 1])
        sdf = skrobot.sdf.trimesh2sdf(mesh)
        assert isinstance(sdf, BoxSDF)

        # primitive mesh (sphere)
        mesh = trimesh.creation.icosphere(subdivisions=3)
        sdf = skrobot.sdf.trimesh2sdf(mesh)
        assert isinstance(sdf, SphereSDF)

        # primitive mesh (cylinder)
        mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
        sdf = skrobot.sdf.trimesh2sdf(mesh)
        assert isinstance(sdf, CylinderSDF)
