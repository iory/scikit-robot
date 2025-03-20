import importlib
import pkg_resources

__version__ = pkg_resources.get_distribution('scikit-robot').version

_SUBMODULES = [
    "coordinates",
    "data",
    "interpolator",
    "planner",
    "interfaces",
    "model",
    "models",
    "viewers",
    "utils",
    "sdf"
]
__all__ = _SUBMODULES


class LazyModule(object):
    def __init__(self, name):
        self.__name__ = "skrobot." + name
        self._name = name
        self._module = None

    def __getattr__(self, attr):
        if self._module is None:
            self._module = importlib.import_module("skrobot." + self._name)
        return getattr(self._module, attr)

    def __dir__(self):
        if self._module is None:
            return ["__all__"]
        return dir(self._module)


_module_cache = {}
for submodule in _SUBMODULES:
    _module_cache[submodule] = LazyModule(submodule)


def __getattr__(name):
    if name in _SUBMODULES:
        return _module_cache[name]
    raise AttributeError(
        "module {} has no attribute {}".format(__name__, name))


def __dir__():
    return __all__ + ['__version__']
