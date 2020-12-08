import os.path as osp
import shutil
import unittest

import numpy as np

import skrobot
import trimesh


class TestAxis(unittest.TestCase):

    def test_init(self):
        skrobot.models.Axis()

    def from_coords(self):
        coords = skrobot.coordinates.Coordinates()
        skrobot.models.Axis.from_coords(coords)

    def from_cascoords(self):
        cascoords = skrobot.coordinates.CascadedCoords()
        skrobot.models.Axis.from_cascoords(cascoords)


class TestBox(unittest.TestCase):

    def test_init(self):
        skrobot.models.Box(extents=(1, 1, 1))
        skrobot.models.Box(extents=(1, 1, 1), with_sdf=True)

    def test_init_with_sdf(self):
        b = skrobot.models.Box(extents=(1, 1, 1), with_sdf=True)
        booleans, _ = b.sdf.on_surface(b.visual_mesh.vertices)
        is_all_vertices_on_surface = np.all(booleans)
        self.assertTrue(is_all_vertices_on_surface)


class TestCone(unittest.TestCase):

    def test_init(self):
        skrobot.models.Cone(radius=0.5, height=1)


class TestCylinder(unittest.TestCase):

    def test_init(self):
        skrobot.models.Cylinder(radius=0.5, height=1)


class TestSphere(unittest.TestCase):

    def test_init(self):
        skrobot.models.Sphere(radius=1)

    def test_init_with_sdf(self):
        s = skrobot.models.Sphere(radius=1.0, with_sdf=True)
        booleans, _ = s.sdf.on_surface(s.visual_mesh.vertices)
        is_all_vertices_on_surface = np.all(booleans)
        self.assertTrue(is_all_vertices_on_surface)


class TestAnnulus(unittest.TestCase):

    def test_init(self):
        skrobot.models.Annulus(r_min=0.2, r_max=0.5, height=1)


class TestMeshLink(unittest.TestCase):

    def test_init(self):
        cylinder = trimesh.creation.cylinder(radius=1.0, height=1.0)
        skrobot.models.MeshLink(cylinder)
        skrobot.models.MeshLink([cylinder, cylinder])

        base_obj_path = osp.join(osp.dirname(skrobot.data.pr2_urdfpath()),
                                 'meshes', 'base_v0', 'base.obj')
        skrobot.models.MeshLink(base_obj_path)

    def test_init_with_sdf(self):
        home_dir = osp.expanduser("~")
        sdf_cache_dir = osp.join(home_dir, '.skrobot', 'sdf')
        if osp.exists(sdf_cache_dir):
            shutil.rmtree(sdf_cache_dir)

        bunny_obj_path = skrobot.data.bunny_objpath()
        m = skrobot.models.MeshLink(bunny_obj_path, with_sdf=True, dim_grid=50)

        booleans, _ = m.sdf.on_surface(m.visual_mesh.vertices)
        is_all_vertices_on_surface = np.all(booleans)
        self.assertTrue(is_all_vertices_on_surface)
