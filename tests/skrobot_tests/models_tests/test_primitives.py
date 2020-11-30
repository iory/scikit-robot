import os.path as osp
import unittest

import skrobot
import trimesh


class TestAxis(unittest.TestCase):

    def test_init(self):
        skrobot.primitives.Axis()

    def from_coords(self):
        coords = skrobot.coordinates.Coordinates()
        skrobot.primitives.Axis.from_coords(coords)

    def from_cascoords(self):
        cascoords = skrobot.coordinates.CascadedCoords()
        skrobot.primitives.Axis.from_cascoords(cascoords)


class TestBox(unittest.TestCase):

    def test_init(self):
        skrobot.primitives.Box(extents=(1, 1, 1))


class TestCone(unittest.TestCase):

    def test_init(self):
        skrobot.primitives.Cone(radius=0.5, height=1)


class TestCylinder(unittest.TestCase):

    def test_init(self):
        skrobot.primitives.Cylinder(radius=0.5, height=1)


class TestSphere(unittest.TestCase):

    def test_init(self):
        skrobot.primitives.Sphere(radius=1)


class TestAnnulus(unittest.TestCase):

    def test_init(self):
        skrobot.primitives.Annulus(r_min=0.2, r_max=0.5, height=1)


class TestMeshLink(unittest.TestCase):

    def test_init(self):
        cylinder = trimesh.creation.cylinder(radius=1.0, height=1.0)
        skrobot.primitives.MeshLink(cylinder)
        skrobot.primitives.MeshLink([cylinder, cylinder])

        base_obj_path = osp.join(osp.dirname(skrobot.data.pr2_urdfpath()),
                                 'meshes', 'base_v0', 'base.obj')
        skrobot.primitives.MeshLink(base_obj_path)
