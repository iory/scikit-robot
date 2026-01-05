import os
import sys
import unittest
import warnings

from skrobot.data import fetch_urdfpath
from skrobot.data import panda_urdfpath
from skrobot.model import RobotModel
from skrobot.models import Fetch
from skrobot.models import Panda
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


class TestRobotModelFromURDFDeprecation(unittest.TestCase):

    def test_deprecation_warning(self):
        """Test that RobotModelFromURDF raises DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RobotModelFromURDF(urdf_file=panda_urdfpath())
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("deprecated", str(w[0].message).lower())


class TestRobotModelURDFParameters(unittest.TestCase):

    def test_robot_model_with_urdf_file_path(self):
        """Test RobotModel initialization with urdf file path."""
        robot = RobotModel(urdf=panda_urdfpath())
        self.assertEqual(robot.name, "panda")

    def test_robot_model_with_urdf_string(self):
        """Test RobotModel initialization with urdf string."""
        with open(panda_urdfpath(), 'r') as f:
            urdf_string = f.read()
        robot = RobotModel(urdf=urdf_string)
        self.assertEqual(robot.name, "panda")

    def test_robot_model_from_urdf_static_method(self):
        """Test RobotModel.from_urdf static method."""
        robot = RobotModel.from_urdf(panda_urdfpath())
        self.assertEqual(robot.name, "panda")

    def test_panda_with_custom_urdf(self):
        """Test Panda with custom urdf parameter."""
        panda = Panda(urdf=panda_urdfpath())
        self.assertEqual(panda.name, "panda")

    def test_panda_with_custom_urdf_file(self):
        """Test Panda with custom urdf_file parameter."""
        panda = Panda(urdf_file=panda_urdfpath())
        self.assertEqual(panda.name, "panda")

    def test_fetch_with_custom_urdf(self):
        """Test Fetch with custom urdf parameter."""
        fetch = Fetch(urdf=fetch_urdfpath())
        self.assertEqual(fetch.name, "fetch")

    def test_panda_raises_error_with_both_urdf_params(self):
        """Test that providing both urdf and urdf_file raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Panda(urdf="test", urdf_file="test2")
        self.assertIn("cannot be given at the same time", str(context.exception))

    def test_fetch_raises_error_with_both_urdf_params(self):
        """Test that providing both urdf and urdf_file raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Fetch(urdf="test", urdf_file="test2")
        self.assertIn("cannot be given at the same time", str(context.exception))
