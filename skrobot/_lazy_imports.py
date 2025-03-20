import importlib


_trimesh = None


def _lazy_trimesh():
    global _trimesh
    if _trimesh is None:
        import trimesh
        _trimesh = trimesh
    return _trimesh


_scipy = None


def _lazy_scipy():
    global _scipy
    if _scipy is None:
        import scipy
        _scipy = scipy
    return _scipy


class LazyImportClass(object):

    def __init__(self, module_name, class_name, package):
        self._module_name = module_name
        self._class_name = class_name
        self._class = None
        self._package = package

    def __getattr__(self, attr):
        if self._class is None:
            try:
                module = importlib.import_module(self._module_name,
                                                 package=self._package)
                self._class = getattr(module, self._class_name)
            except ImportError:
                self._class = None
        if self._class is None:
            raise AttributeError("Failed to load {}".format(self._class_name))
        return getattr(self._class, attr)

    def __call__(self, *args, **kwargs):
        if self._class is None:
            try:
                module = importlib.import_module(self._module_name,
                                                 package=self._package)
                self._class = getattr(module, self._class_name)
            except ImportError as e:
                raise AttributeError(
                    "Failed to load {}. Error log: {}".format(
                        self._class_name, e))
        return self._class(*args, **kwargs)

    def __dir__(self):
        if self._class is None:
            return []
        return dir(self._class)
