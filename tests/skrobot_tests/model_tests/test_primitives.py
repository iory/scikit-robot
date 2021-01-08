import os.path as osp
import shutil
import unittest

import numpy as np
import trimesh

import skrobot
from skrobot.data import get_cache_dir


class TestAxis(unittest.TestCase):

    def test_init(self):
        skrobot.model.Axis()

    def from_coords(self):
        coords = skrobot.coordinates.Coordinates()
        skrobot.model.Axis.from_coords(coords)

    def from_cascoords(self):
        cascoords = skrobot.coordinates.CascadedCoords()
        skrobot.model.Axis.from_cascoords(cascoords)


class TestBox(unittest.TestCase):

    def test_init(self):
        skrobot.model.Box(extents=(1, 1, 1))
        skrobot.model.Box(extents=(1, 1, 1), with_sdf=True)

    def test_init_with_sdf(self):
        b = skrobot.model.Box(extents=(1, 1, 1), with_sdf=True)
        booleans, _ = b.sdf.on_surface(b.visual_mesh.vertices)
        is_all_vertices_on_surface = np.all(booleans)
        self.assertTrue(is_all_vertices_on_surface)


class TestCone(unittest.TestCase):

    def test_init(self):
        skrobot.model.Cone(radius=0.5, height=1)


class TestCylinder(unittest.TestCase):

    def test_init(self):
        skrobot.model.Cylinder(radius=0.5, height=1)


class TestSphere(unittest.TestCase):

    def test_init(self):
        skrobot.model.Sphere(radius=1)

    def test_init_with_sdf(self):
        s = skrobot.model.Sphere(radius=1.0, with_sdf=True)
        booleans, _ = s.sdf.on_surface(s.visual_mesh.vertices)
        is_all_vertices_on_surface = np.all(booleans)
        self.assertTrue(is_all_vertices_on_surface)


class TestAnnulus(unittest.TestCase):

    def test_init(self):
        skrobot.model.Annulus(r_min=0.2, r_max=0.5, height=1)


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

        # test for trimesh.base.Trimeh
        bunny_obj_path = skrobot.data.bunny_objpath()
        m = skrobot.model.MeshLink(
            trimesh.load(bunny_obj_path), with_sdf=True, dim_grid=50)

        bunny_obj_path = skrobot.data.bunny_objpath()
        m = skrobot.model.MeshLink(bunny_obj_path, with_sdf=True, dim_grid=50)

        booleans, _ = m.sdf.on_surface(m.visual_mesh.vertices)
        is_all_vertices_on_surface = np.all(booleans)
        self.assertTrue(is_all_vertices_on_surface)
