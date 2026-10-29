"""A URDF ``<material>``'s ``<texture>`` on the link's visual mesh."""

import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import PIL.Image
import trimesh

from skrobot.models.urdf import RobotModelFromURDF


# A unit square whose left edge has u = 0 and right edge u = 1, its
# material a plain grey with no image (as PR2's wheel.obj)
QUAD_MTL = """newmtl grey
Kd 0.64 0.64 0.64
"""
QUAD_OBJ = """mtllib quad.mtl
usemtl grey
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3
f 1/1 3/3 4/4
"""


def urdf(geometry, material):
    return """<?xml version="1.0"?>
<robot name="textured">
  <link name="base">
    <visual>
      <geometry>{}</geometry>
      <material name="m">{}</material>
    </visual>
  </link>
</robot>""".format(geometry, material)


class TestURDFTexture(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        # left half red, right half blue
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :2] = [255, 0, 0]
        image[:, 2:] = [0, 0, 255]
        PIL.Image.fromarray(image).save(os.path.join(self.dir, 'rb.png'))
        with open(os.path.join(self.dir, 'quad.mtl'), 'w') as f:
            f.write(QUAD_MTL)
        with open(os.path.join(self.dir, 'quad.obj'), 'w') as f:
            f.write(QUAD_OBJ)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def load(self, geometry, material, warned=()):
        path = os.path.join(self.dir, 'robot.urdf')
        with open(path, 'w') as f:
            f.write(urdf(geometry, material))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            robot = RobotModelFromURDF(urdf_file=path)
        self.assertEqual(
            list(warned), [str(w.message) for w in caught
                           if 'texture' in str(w.message)])
        mesh = robot.base.visual_mesh
        return mesh[0] if isinstance(mesh, list) else mesh

    def test_laid_on_uvs(self):
        mesh = self.load('<mesh filename="quad.obj"/>',
                         '<texture filename="rb.png"/>')
        colors = mesh.visual.vertex_colors[:, :3]
        u = mesh.vertices[:, 0]
        np.testing.assert_array_equal(colors[u == 0], [[255, 0, 0]] * 2)
        np.testing.assert_array_equal(colors[u == 1], [[0, 0, 255]] * 2)

    def test_without_uvs_untouched(self):
        mesh = self.load('<box size="1 1 1"/>',
                         '<texture filename="rb.png"/>')
        np.testing.assert_array_equal(
            mesh.visual.face_colors,
            [trimesh.visual.DEFAULT_COLOR] * 12)

    def test_image_not_found_warned(self):
        mesh = self.load('<mesh filename="quad.obj"/>',
                         '<texture filename="missing.png"/>',
                         warned=['texture image missing.png '
                                 'could not be loaded'])
        self.assertEqual(
            1, len(np.unique(mesh.visual.vertex_colors, axis=0)))

    def test_without_uvs_the_colour_given(self):
        mesh = self.load('<box size="1 1 1"/>',
                         '<color rgba="0 1 0 1"/>'
                         '<texture filename="rb.png"/>')
        np.testing.assert_array_equal(
            mesh.visual.face_colors, [[0, 255, 0, 255]] * 12)
