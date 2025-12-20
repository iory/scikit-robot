# flake8: noqa

import sys
import warnings


def _check_correct_package_installed():
    """Check if scikit-robot is correctly installed instead of wrong 'skrobot' package."""
    try:
        if sys.version_info >= (3, 7):
            import importlib.metadata
            try:
                importlib.metadata.version('scikit-robot')
                return True
            except importlib.metadata.PackageNotFoundError:
                pass
        else:
            import pkg_resources
            try:
                pkg_resources.get_distribution('scikit-robot')
                return True
            except pkg_resources.DistributionNotFound:
                pass
    except Exception:
        return True

    warnings.warn(
        "\n"
        "=" * 70 + "\n"
        "WARNING: Wrong 'skrobot' package detected!\n"
        "=" * 70 + "\n"
        "It appears you have installed the wrong package.\n"
        "You may have run 'pip install skrobot' instead of 'pip install scikit-robot'.\n\n"
        "To fix this, please run:\n"
        "    pip uninstall skrobot\n"
        "    pip install scikit-robot\n"
        "=" * 70,
        UserWarning,
        stacklevel=2
    )
    return False


_check_correct_package_installed()


if (sys.version_info[0] == 3 and sys.version_info[1] >= 7) \
    or sys.version_info[0] > 3:
    import importlib
    import importlib.metadata
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
        "sdf",
        "urdf"
    ]
    __all__ = _SUBMODULES
    _version = None

    def determine_version(module_name):
        return importlib.metadata.version(module_name)


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
                _version = determine_version('scikit-robot')
            return _version
        if name in _SUBMODULES:
            return _module_cache[name]
        raise AttributeError(
            "module {} has no attribute {}".format(__name__, name))


    def __dir__():
        return __all__ + ['__version__']
else:
    import pkg_resources


    def determine_version(module_name):
        return pkg_resources.get_distribution(module_name).version

    __version__ = determine_version('scikit-robot')


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
    from skrobot import urdf
