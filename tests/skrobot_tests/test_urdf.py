import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

from skrobot.coordinates.math import rpy_angle
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

        for attr in ['collision_mesh', 'visual_mesh']:
            mesh = getattr(dummy_robot.base_link, attr)
            if isinstance(mesh, list):
                origin = mesh[0].metadata["origin"]
            else:
                origin = mesh.metadata["origin"]
            rot = origin[:3, :3]
            determinant = np.linalg.det(rot)
            self.assertAlmostEqual(determinant, 1.0, places=5)
            rpy = rpy_angle(rot)[0][::-1]
            self.assertTrue(np.allclose(rpy, [0.1, 0.2, 0.3]))

            trans = origin[:3, 3]
            self.assertTrue(np.allclose(trans, [10, 20, 30]))
