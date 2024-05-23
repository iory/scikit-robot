import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

from skrobot.data import bunny_objpath
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

    def test_load_urdfmodel_with_scale_parameter(self):
        td = tempfile.mkdtemp()
        urdf_file = """
        <robot name="myfirst">
          <link name="base_link">
            <visual>
              <geometry>
                <mesh filename="./bunny.obj" scale="10 10 10" />
              </geometry>
            </visual>
            <collision>
              <geometry>
                <mesh filename="./bunny.obj" scale="10 10 10" />
              </geometry>
            </collision>
          </link>
        </robot>
        """
        # write urdf file
        with open(os.path.join(td, 'temp.urdf'), 'w') as f:
            f.write(urdf_file)

        shutil.copy(bunny_objpath(), os.path.join(td, 'bunny.obj'))
        urdf_file = os.path.join(td, 'temp.urdf')
        dummy_robot = RobotModelFromURDF(urdf_file=urdf_file)
        origin = dummy_robot.base_link.collision_mesh.metadata["origin"]
        rot = origin[:3, :3]
        determinant = np.linalg.det(rot)
        self.assertAlmostEqual(determinant, 1.0, places=5)
        # TODO(HiroIshida): check if trans is correctly scaled
        # however, to check this, we must fix the issue:
        # https://github.com/mmatl/urdfpy/issues/17
