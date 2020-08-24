import os.path as osp
import unittest

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


class TestCone(unittest.TestCase):

    def test_init(self):
        skrobot.models.Cone(radius=0.5, height=1)


class TestCylinder(unittest.TestCase):

    def test_init(self):
        skrobot.models.Cylinder(radius=0.5, height=1)


class TestSphere(unittest.TestCase):

    def test_init(self):
        skrobot.models.Sphere(radius=1)


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
