import unittest

import skrobot


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


class TestCylinder(unittest.TestCase):

    def test_init(self):
        skrobot.models.Cylinder(radius=0.5, height=1)


class TestSphere(unittest.TestCase):

    def test_init(self):
        skrobot.models.Sphere(radius=1)


class TestAnnulus(unittest.TestCase):

    def test_init(self):
        skrobot.models.Annulus(r_min=0.2, r_max=0.5, height=1)
