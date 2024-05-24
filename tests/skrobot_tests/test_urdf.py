import os
import shutil
import sys
import tempfile
import trimesh
import unittest

import numpy as np

from skrobot.coordinates.math import rpy_angle
from skrobot.coordinates.math import rpy_matrix
from skrobot.data import bunny_objpath
from skrobot.data import fetch_urdfpath
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.utils.urdf import mesh_simplify_factor


class TestURDF(unittest.TestCase):

    def setUp(self):
        td = tempfile.mkdtemp()
        urdf_file = """
        <robot name="myfirst">
          <link name="base_link">
            <visual>
              <geometry>
                <mesh filename="./bunny.obj" scale="10 10 10" />
                <origin rpy="0.1 0.2 0.3" xyz="1 2 3" />
              </geometry>
            </visual>
            <collision>
              <geometry>
                <mesh filename="./bunny.obj" scale="10 10 10" />
                <origin rpy="0.1 0.2 0.3" xyz="1 2 3" />
              </geometry>
            </collision>
          </link>
        </robot>
        """
        # write urdf file
        with open(os.path.join(td, 'temp.urdf'), 'w') as f:
            f.write(urdf_file)

        shutil.copy(bunny_objpath(), os.path.join(td, 'bunny.obj'))
        self.temp_urdf_dir = td

    def tearDown(self):
        shutil.rmtree(self.temp_urdf_dir)

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

    def test_load_urdfmodel_origin_parse(self):
        urdf_file = os.path.join(self.temp_urdf_dir, 'temp.urdf')
        dummy_robot = RobotModelFromURDF(urdf_file=urdf_file)

        # create ground truth bunny vertices
        bunny_mesh = trimesh.load_mesh(bunny_objpath())
        bunny_verts = bunny_mesh.vertices
        bunny_verts_scaled = bunny_verts * 10
        rotmat = rpy_matrix(0.3, 0.2, 0.1)  # skrobot uses reversed order
        trans = np.array([1., 2., 3.])
        bunny_verts_deformed_gt = np.dot(bunny_verts_scaled, rotmat.T) + trans

        for attr in ['visual_mesh', 'collision_mesh', 'visual_mesh']:
            mesh = getattr(dummy_robot.base_link, attr)
            if isinstance(mesh, list):
                mesh =mesh[0]
            self.assertTrue(np.allclose(mesh.vertices, bunny_verts_deformed_gt))

            # origin must be np.eye(4) because the transformation is already applied
            origin = mesh.metadata['origin']
            self.assertTrue(np.allclose(origin, np.eye(4)))
