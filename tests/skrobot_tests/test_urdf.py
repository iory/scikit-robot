import os
import shutil
import sys
import tempfile
import unittest

import trimesh

from skrobot.data import fetch_urdfpath
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.utils.urdf import mesh_simplify_factor


class TestURDF(unittest.TestCase):

    def test_load_urdfmodel(self):
        urdfpath = fetch_urdfpath()
        # Absolute path
        RobotModelFromURDF(urdf_file=urdfpath)
        # Relative path
        os.chdir(os.path.dirname(urdfpath))
        RobotModelFromURDF(
            urdf_file=os.path.basename(urdfpath))
        # String
        with open(urdfpath, 'r') as f:
            RobotModelFromURDF(urdf=f.read())

    def test_load_urdfmodel_with_simplification(self):
        if sys.version_info.major < 3:
            return  # this feature is supported only for python3.x

        # create cache and load
        with mesh_simplify_factor(0.1):
            RobotModelFromURDF(urdf_file=fetch_urdfpath())

        # load using existing cache
        with mesh_simplify_factor(0.1):
            RobotModelFromURDF(urdf_file=fetch_urdfpath())

    def test_mirror_scale_keeps_outward_normals(self):
        # A negative scale (e.g. ``1 -1 1`` used to mirror a mesh) reverses
        # the triangle winding order. The visual mesh must still have its
        # normals pointing outward, otherwise backface culling makes the
        # model look see-through. A watertight box has a positive signed
        # volume only when its faces are wound outward, so we use the sign
        # of the volume as a robust check.
        tmpdir = tempfile.mkdtemp()
        try:
            mesh_path = os.path.join(tmpdir, 'box.stl')
            box = trimesh.creation.box(extents=(0.2, 0.1, 0.05))
            self.assertGreater(box.volume, 0)
            box.export(mesh_path)

            urdf_path = os.path.join(tmpdir, 'mirror.urdf')
            with open(urdf_path, 'w') as f:
                f.write(
                    '<?xml version="1.0"?>\n'
                    '<robot name="mirror_test">\n'
                    '  <link name="base_link">\n'
                    '    <visual>\n'
                    '      <geometry>\n'
                    '        <mesh filename="box.stl" scale="1 -1 1"/>\n'
                    '      </geometry>\n'
                    '    </visual>\n'
                    '  </link>\n'
                    '</robot>\n')

            robot = RobotModelFromURDF(urdf_file=urdf_path)
            visual_mesh = robot.base_link.visual_mesh
            mesh = visual_mesh[0] if isinstance(visual_mesh, list) \
                else visual_mesh

            # mirrored across Y ...
            self.assertLess(mesh.vertices[:, 1].min(), 0)
            self.assertGreater(mesh.vertices[:, 1].max(), 0)
            # ... but still wound outward (positive signed volume).
            self.assertGreater(mesh.volume, 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
