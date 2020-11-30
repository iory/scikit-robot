import os.path as osp
import unittest

import skrobot
import trimesh


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


class TestCone(unittest.TestCase):

    def test_init(self):
        skrobot.model.Cone(radius=0.5, height=1)


class TestCylinder(unittest.TestCase):

    def test_init(self):
        skrobot.model.Cylinder(radius=0.5, height=1)


class TestSphere(unittest.TestCase):

    def test_init(self):
        skrobot.model.Sphere(radius=1)


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
