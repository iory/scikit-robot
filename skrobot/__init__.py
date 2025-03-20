import importlib


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
_pkg_resources = None
_version = None


def _lazy_pkg_resources():
    global _pkg_resources
    if _pkg_resources is None:
        import pkg_resources
        _pkg_resources = pkg_resources
    return _pkg_resources


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
    global _version
    if name == "__version__":
        if _version is None:
            pkg_resources = _lazy_pkg_resources()
            _version = pkg_resources.get_distribution(
                'scikit-robot').version
        return _version
    if name in _SUBMODULES:
        return _module_cache[name]
    raise AttributeError(
        "module {} has no attribute {}".format(__name__, name))


def __dir__():
    return __all__ + ['__version__']
