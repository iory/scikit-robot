# flake8: noqa

import importlib


class LazyPybulletModule(object):
    def __init__(self):
        self._module = None

    def __getattr__(self, attr):
        if self._module is None:
            self._module = importlib.import_module('skrobot.interfaces._pybullet')
        return getattr(self._module, attr)

    def __dir__(self):
        if self._module is None:
            return []
        return dir(self._module)


# Lazy import for pybullet module
pybullet = LazyPybulletModule()


def __getattr__(name):
    if name == 'PybulletRobotInterface':
        return pybullet.PybulletRobotInterface
    elif name == '_pybullet':
        # Force lazy loading - only import when explicitly requested
        if pybullet._module is None:
            pybullet._module = importlib.import_module('skrobot.interfaces._pybullet')
        return pybullet._module
    raise AttributeError(f"module 'skrobot.interfaces' has no attribute '{name}'")


from .ros import *
