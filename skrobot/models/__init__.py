# flake8: noqa

# a workaround for preserving the backward-compatibility
# so that a user also can import primitive by
# import skrobot.models.primitive 
from skrobot.model.primitives import *

from skrobot.models.fetch import Fetch
from skrobot.models.kuka import Kuka
from skrobot.models.panda import Panda
from skrobot.models.pr2 import PR2
