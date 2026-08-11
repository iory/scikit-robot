import os
import shutil
import tempfile
import unittest

import numpy as np
import trimesh

from skrobot.model import RobotModel


URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="normals">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="{rpy}"/>
      <geometry>
        <mesh filename="{mesh}"{scale}/>
      </geometry>
    </visual>
  </link>
</robot>
"""


class TestVisualMeshKeepsAuthoredNormals(unittest.TestCase):
    """Normals stored in a mesh file must reach ``Link.visual_mesh``.

    They are the only record of which edges the author meant to be smooth.
    Once dropped, the only thing left is to average adjacent face normals,
    which rounds off every hard edge -- so any renderer downstream shows a
    flat plate as a gradient, or a cylinder as a faceted tube.
    """

    #: Not a normal any averaging would produce for a box, so its presence
    #: proves the array came from the file rather than from inference.
    AUTHORED = np.array([0.0, 0.0, 1.0])

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        mesh = trimesh.creation.box()
        mesh.vertex_normals = np.tile(self.AUTHORED, (len(mesh.vertices), 1))
        self.mesh_path = os.path.join(self.tmpdir, 'box.glb')
        mesh.export(self.mesh_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _load(self, rpy='0 0 0', scale=None):
        urdf_path = os.path.join(self.tmpdir, 'robot.urdf')
        with open(urdf_path, 'w') as f:
            f.write(URDF_TEMPLATE.format(
                mesh=self.mesh_path, rpy=rpy,
                scale='' if scale is None else ' scale="{}"'.format(scale)))
        robot = RobotModel()
        with open(urdf_path) as f:
            robot.load_urdf_file(f)
        link = robot.link_list[0]
        self.assertTrue(link.visual_mesh)
        return link.visual_mesh[0]

    def test_they_are_still_there_after_loading(self):
        mesh = self._load()
        self.assertIn('vertex_normals', mesh._cache.cache)
        np.testing.assert_allclose(
            np.asarray(mesh.vertex_normals),
            np.tile(self.AUTHORED, (len(mesh.vertices), 1)), atol=1e-6)

    def test_a_visual_origin_rotates_them(self):
        # quarter turn about x: +z becomes -y
        mesh = self._load(rpy='{} 0 0'.format(np.pi / 2))
        np.testing.assert_allclose(
            np.asarray(mesh.vertex_normals),
            np.tile([0.0, -1.0, 0.0], (len(mesh.vertices), 1)), atol=1e-6)

    def test_a_mirroring_scale_keeps_them_on_the_outside(self):
        """``scale="1 -1 1"`` reverses the winding; the normals have to be
        negated to match, or the surface renders inside out."""
        mesh = self._load(scale='1 -1 1')
        np.testing.assert_allclose(
            np.asarray(mesh.vertex_normals),
            np.tile([0.0, 0.0, -1.0], (len(mesh.vertices), 1)), atol=1e-6)

    def test_a_non_uniform_scale_maps_them_by_the_inverse_transpose(self):
        """Squashing z leaves a +z normal pointing at +z, and unit length.

        Checking the direction as well as the length matters: recomputed box
        normals would also come out unit length, so length alone would pass
        even with the normals thrown away.
        """
        mesh = self._load(scale='1 1 0.1')
        np.testing.assert_allclose(
            np.asarray(mesh.vertex_normals),
            np.tile(self.AUTHORED, (len(mesh.vertices), 1)), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
