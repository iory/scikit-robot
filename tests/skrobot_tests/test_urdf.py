import sys
import unittest

from skrobot.data import fetch_urdfpath
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.utils.urdf import mesh_simplify_factor


class TestURDF(unittest.TestCase):

    def test_load_urdfmodel(self):
        RobotModelFromURDF(urdf_file=fetch_urdfpath())

    def test_load_urdfmodel_with_simplification(self):
        if sys.version_info.major < 3:
            return  # this feature is supported only for python3.x

        # create cache and load
        with mesh_simplify_factor(0.1):
            RobotModelFromURDF(urdf_file=fetch_urdfpath())

        # load using existing cache
        with mesh_simplify_factor(0.1):
            RobotModelFromURDF(urdf_file=fetch_urdfpath())
