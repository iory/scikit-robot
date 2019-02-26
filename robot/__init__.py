import pkg_resources


__version__ = pkg_resources.get_distribution('robot').version


from robot import coordinates

worldcoords = coordinates.CascadedCoords()

from robot import geo  # NOQA
from robot import interpolator  # NOQA
from robot import math  # NOQA
from robot import optimizer  # NOQA
from robot import optimizers  # NOQA
from robot import robot_model  # NOQA
from robot import robot_models  # NOQA
from robot import symbol_math  # NOQA
from robot import utils  # NOQA
from robot.pybullet_robot_interface import PybulletRobotInterface  # NOQA
from robot.utils.urdf import URDF  # NOQA
from robot import quaternion  # NOQA
from robot import data
