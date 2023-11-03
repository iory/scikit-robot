# flake8: noqa

import pkg_resources


__version__ = pkg_resources.get_distribution('scikit-robot').version


from skrobot import coordinates
from skrobot import data
from skrobot import interpolator
from skrobot import planner
from skrobot import interfaces
from skrobot import model
from skrobot import models
from skrobot import viewers
from skrobot import utils
from skrobot import sdf
