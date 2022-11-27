import copy
import os.path as osp
import pickle
import shutil
import unittest

import numpy as np
import trimesh
from trimesh import PointCloud

import skrobot
from skrobot.data import get_cache_dir


class TestAxis(unittest.TestCase):

    def test_init(self):
        prim = skrobot.model.Axis()
        assert prim._visual_mesh is not None
        assert prim._collision_mesh is None

    def from_coords(self):
        coords = skrobot.coordinates.Coordinates()
        skrobot.model.Axis.from_coords(coords)

    def from_cascoords(self):
        cascoords = skrobot.coordinates.CascadedCoords()
        skrobot.model.Axis.from_cascoords(cascoords)


class TestBox(unittest.TestCase):

    def test_init(self):
        prim = skrobot.model.Box(extents=(1, 1, 1))
        assert prim._visual_mesh is not None
        assert prim._collision_mesh is not None

    def test_init_with_sdf(self):
        pos = np.ones(3)
        b = skrobot.model.Box(extents=(1, 1, 1), pos=pos, with_sdf=True)
        booleans, _ = b.sdf.on_surface(b.visual_mesh.vertices + pos)
        is_all_vertices_on_surface = np.all(booleans)
        self.assertTrue(is_all_vertices_on_surface)


class TestCone(unittest.TestCase):

    def test_init(self):
        prim = skrobot.model.Cone(radius=0.5, height=1)
        assert prim._visual_mesh is not None
        assert prim._collision_mesh is not None


class TestCylinder(unittest.TestCase):

    def test_init(self):
        prim = skrobot.model.Cylinder(radius=0.5, height=1)
        assert prim._visual_mesh is not None
        assert prim._collision_mesh is not None


class TestSphere(unittest.TestCase):

    def test_init(self):
        prim = skrobot.model.Sphere(radius=1)
        assert prim._visual_mesh is not None
        assert prim._collision_mesh is not None

    def test_init_with_sdf(self):
        pos = np.ones(3)
        s = skrobot.model.Sphere(radius=1.0, pos=pos, with_sdf=True)
        booleans, _ = s.sdf.on_surface(s.visual_mesh.vertices + pos)
        is_all_vertices_on_surface = np.all(booleans)
        self.assertTrue(is_all_vertices_on_surface)


class TestAnnulus(unittest.TestCase):

    def test_init(self):
        prim = skrobot.model.Annulus(r_min=0.2, r_max=0.5, height=1)
        assert prim._visual_mesh is not None
        assert prim._collision_mesh is not None


class TestLineString(unittest.TestCase):

    def test_init(self):
        points = np.random.randn(100, 3)
        skrobot.model.LineString(points)

        with self.assertRaises(AssertionError):
            # array dimension must be 2
            skrobot.model.LineString(np.random.randn(100,))
        with self.assertRaises(AssertionError):
            # point dimension must be 3
            skrobot.model.LineString(np.random.randn(100, 1))
        with self.assertRaises(AssertionError):
            # len(paths) must be > 1
            skrobot.model.LineString(np.random.randn(1, 3))


class TestMeshLink(unittest.TestCase):

    def test_init(self):
        cylinder = trimesh.creation.cylinder(radius=1.0, height=1.0)
        skrobot.model.MeshLink(cylinder)
        skrobot.model.MeshLink([cylinder, cylinder])

        base_obj_path = osp.join(osp.dirname(skrobot.data.pr2_urdfpath()),
                                 'meshes', 'base_v0', 'base.obj')
        skrobot.model.MeshLink(base_obj_path)

    def test_init_with_sdf(self):
        sdf_cache_dir = osp.join(get_cache_dir(), 'sdf')
        if osp.exists(sdf_cache_dir):
            shutil.rmtree(sdf_cache_dir)

        # test: input is trimesh
        bunny_obj_path = skrobot.data.bunny_objpath()
        m = skrobot.model.MeshLink(
            trimesh.load(bunny_obj_path), with_sdf=True, dim_grid=50)

        # test: input is mesh path
        pos = np.ones(3)
        bunny_obj_path = skrobot.data.bunny_objpath()
        m = skrobot.model.MeshLink(
            bunny_obj_path, pos=pos, with_sdf=True, dim_grid=50)

        booleans, _ = m.sdf.on_surface(m.visual_mesh.vertices + pos)
        is_all_vertices_on_surface = np.all(booleans)
        self.assertTrue(is_all_vertices_on_surface)


class TestPointCloudLink(unittest.TestCase):

    def test_init_(self):
        with self.assertRaises(AssertionError):
            skrobot.model.PointCloudLink(np.random.randn(100,))
        with self.assertRaises(AssertionError):
            skrobot.model.PointCloudLink(np.random.randn(100, 1))

        pts = np.random.randn(100, 3)
        skrobot.model.PointCloudLink(pts)

        pts_mesh = PointCloud(pts)
        skrobot.model.PointCloudLink(pts_mesh)


class TestPrimitiveSerialization(unittest.TestCase):

    @staticmethod
    def check_mesh(mesh_like1, mesh_like2):
        # TODO(HiroIshida): pickling and compare bytes seems easy
        # bit somehow serialized value does not match
        # so do the following check

        if isinstance(mesh_like1, list):
            assert len(mesh_like1) == len(mesh_like2)
            for mesh1, mesh2 in zip(mesh_like1, mesh_like2):
                verts1 = mesh1.vertices
                verts2 = mesh2.vertices
                np.testing.assert_almost_equal(verts1, verts2)
        else:
            verts1 = mesh_like1.vertices
            verts2 = mesh_like2.vertices
            np.testing.assert_almost_equal(verts1, verts2)

    def test_light_serialization(self):
        primitives = [
            skrobot.model.Axis(),
            skrobot.model.Box([1., 1., 1.]),
            skrobot.model.CameraMarker(),
            skrobot.model.Cone(1., 1.),
            skrobot.model.Cylinder(1., 1.),
            skrobot.model.Sphere(1.),
            skrobot.model.Annulus(1., 1., 1.)]

        primitives_ls = copy.deepcopy(primitives)
        for p in primitives_ls:
            p._light_serialize = True

        # test serialization size
        for prim, prim_ls in zip(primitives, primitives_ls):
            byte = pickle.dumps(prim)
            byte_ls = pickle.dumps(prim_ls)
            assert len(byte_ls) < len(byte) * 0.5

        # test serialization consistency
        # NOTE: same mesh means same attributes
        for prim, prim_ls in zip(primitives, primitives_ls):
            prim_again = pickle.loads(pickle.dumps(prim))
            prim_ls_again = pickle.loads(pickle.dumps(prim_ls))

            # collision mesh maybe None
            if prim._collision_mesh is None:
                assert prim_ls._collision_mesh is None
            else:
                self.check_mesh(
                    prim_again._collision_mesh, prim_ls_again._collision_mesh)

            # visual mesh is supposed to not None
            self.check_mesh(
                prim_again._visual_mesh, prim_ls_again._visual_mesh)
