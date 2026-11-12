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
from skrobot.utils.urdf import keep_texture


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
        self.robot = robot
        # the colours the viewers draw: textured meshes baked to vertex
        # colours (visual_mesh keeps the texture itself)
        return robot.base.concatenated_visual_mesh

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

    def test_uvs_without_image_the_colour_given(self):
        # UVs but no material image (as PR2's COLLADA meshes): nothing to
        # keep, so the URDF colour applies as before
        bare = trimesh.Trimesh(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 1, 2], [0, 2, 3]], process=False)
        bare.visual = trimesh.visual.TextureVisuals(
            uv=[[0, 0], [1, 0], [1, 1], [0, 1]],
            material=trimesh.visual.material.PBRMaterial())
        bare.export(os.path.join(self.dir, 'bare.glb'))
        self.load('<mesh filename="bare.glb"/>',
                  '<color rgba="0 1 0 1"/>')
        self.assertIsInstance(
            self.visual().visual, trimesh.visual.ColorVisuals)
        np.testing.assert_array_equal(
            self.robot.base.colors, [[0, 255, 0, 255]] * 2)

    def visual(self):
        mesh = self.robot.base.visual_mesh
        return mesh[0] if isinstance(mesh, list) else mesh

    def test_texture_kept_on_the_visual_mesh(self):
        self.load('<mesh filename="quad.obj"/>',
                  '<texture filename="rb.png"/>')
        visual = self.visual().visual
        self.assertIsInstance(visual, trimesh.visual.TextureVisuals)
        self.assertEqual((4, 4), visual.material.image.size)
        np.testing.assert_allclose(
            visual.uv[self.visual().vertices[:, 0] == 1][:, 0], [1, 1])

    def test_keep_texture_false_bakes_on_load(self):
        with keep_texture(False):
            self.load('<mesh filename="quad.obj"/>',
                      '<texture filename="rb.png"/>')
        mesh = self.visual()
        self.assertIsInstance(mesh.visual, trimesh.visual.ColorVisuals)
        colors = mesh.visual.vertex_colors[:, :3]
        np.testing.assert_array_equal(
            colors[mesh.vertices[:, 0] == 1], [[0, 0, 255]] * 2)

    def test_set_color_and_reset(self):
        self.load('<mesh filename="quad.obj"/>',
                  '<texture filename="rb.png"/>')
        link = self.robot.base
        self.assertFalse(link.color_overridden)
        before = np.array(link.colors)
        link.set_color([0, 255, 0, 255])
        self.assertTrue(link.color_overridden)
        np.testing.assert_array_equal(
            link.colors, [[0, 255, 0, 255]] * 2)
        # the textured mesh is not painted
        self.assertIsInstance(
            self.visual().visual, trimesh.visual.TextureVisuals)
        link.reset_color()
        self.assertFalse(link.color_overridden)
        np.testing.assert_array_equal(link.colors, before)

    def test_set_alpha_overrides(self):
        self.load('<mesh filename="quad.obj"/>',
                  '<texture filename="rb.png"/>')
        link = self.robot.base
        link.set_alpha(0.5)
        self.assertTrue(link.color_overridden)
        np.testing.assert_array_equal(link.colors[:, 3], [128, 128])

    def test_one_file_shares_its_images(self):
        # an OBJ whose own material has the image, on two links
        with open(os.path.join(self.dir, 'own.mtl'), 'w') as f:
            f.write('newmtl own\nKd 1 1 1\nmap_Kd rb.png\n')
        with open(os.path.join(self.dir, 'own.obj'), 'w') as f:
            f.write(QUAD_OBJ.replace('quad.mtl', 'own.mtl')
                    .replace('usemtl grey', 'usemtl own'))
        path = os.path.join(self.dir, 'two.urdf')
        with open(path, 'w') as f:
            f.write("""<?xml version="1.0"?>
<robot name="two">
  <link name="a"><visual><geometry>
    <mesh filename="own.obj"/></geometry></visual></link>
  <link name="b"><visual><geometry>
    <mesh filename="own.obj"/></geometry></visual></link>
  <joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint>
</robot>""")
        robot = RobotModelFromURDF(urdf_file=path)
        images = [link.visual_mesh[0].visual.material.image
                  for link in (robot.a, robot.b)]
        self.assertIsNotNone(images[0])
        self.assertIs(images[0], images[1])
