# flake8: noqa

import pkg_resources


__version__ = pkg_resources.get_distribution('skrobot').version


from skrobot import coordinates
from skrobot import data
from skrobot import interpolator
from skrobot import optimizer
from skrobot import optimizers
from skrobot import interfaces
from skrobot import model
from skrobot import models
from skrobot import visualization
from skrobot import utils
