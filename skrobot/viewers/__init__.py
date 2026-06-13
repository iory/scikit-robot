# flake8: noqa

import inspect


def warn_gl(error_log):
    if 'Library "GL" not found.' in str(error_log):
        print(
            '\x1b[31m'  # red
            + 'Library "GL" not found. Please install it by running:\n'
            + 'sudo apt-get install freeglut3-dev'
            + '\x1b[39m'  # reset
        )


class DummyViewer(object):
    def __init__(self, *args, **kwargs):
        self.has_exit = True
        self._is_active = False
    @property
    def is_active(self):
        return self._is_active
    def redraw(self):
        pass
    def show(self):
        pass
    def add(self, *args, **kwargs):
        pass
    def delete(self, *args, **kwargs):
        pass
    def set_camera(self, *args, **kwargs):
        pass
    def save_image(self, file_obj):
        pass
    def close(self):
        self._is_active = False
    def wait_until_close(self, *args, **kwargs):
        pass
    def pause(self, *args, **kwargs):
        pass

try:
    from ._trimesh import TrimeshSceneViewer
except TypeError:
    # trimesh.viewer.SceneViewer can have function type.
    class TrimeshSceneViewer(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('TrimeshSceneViewer cannot be initialized. '
                               'This issue happens when the X window system '
                               'is not running.')
except (ImportError, RuntimeError) as error_log:
    warn_gl(error_log)
    class TrimeshSceneViewer(DummyViewer):
        _import_error = error_log

try:
    from ._pyrender import PyrenderViewer
    import pyrender
    args = inspect.getfullargspec(pyrender.Viewer.__init__).args
    if 'auto_start' not in args:
        class PyrenderViewer(object):
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    'The correct version of pyrender is not installed.\n'
                    'To install the appropriate version of pyrender, '
                    'please execute the following command:\n'
                    'pip uninstall -y pyrender && pip install scikit-robot-pyrender --no-cache-dir'
                )
except (ImportError, RuntimeError) as error_log:
    warn_gl(error_log)
    class PyrenderViewer(DummyViewer):
        _import_error = error_log

try:
    from ._notebook import JupyterNotebookViewer
except ImportError:
    class JupyterNotebookViewer(DummyViewer):
        pass

try:
    from ._viser import ViserViewer
except ImportError as e:
    import importlib

    _viser_spec = importlib.util.find_spec("viser")
    if _viser_spec is not None:
        import warnings

        warnings.warn(
            "viser is installed but failed to import: {}\n"
            "This may be caused by an incompatible version of a dependency "
            "(e.g. imageio < 2.0).\n"
            "Try: pip install -U viser".format(e),
            stacklevel=1,
        )

    class ViserViewer(DummyViewer):
        pass

# Backwards compatibility alias
ViserVisualizer = ViserViewer


# Mapping from viewer name to its class. Used by :func:`create_viewer`.
_VIEWER_CLASSES = {
    'trimesh': TrimeshSceneViewer,
    'pyrender': PyrenderViewer,
    'viser': ViserViewer,
    'notebook': JupyterNotebookViewer,
}


def _supported_kwargs(cls, kwargs):
    """Return the subset of ``kwargs`` accepted by ``cls.__init__``.

    Each viewer backend takes a different set of constructor options
    (e.g. ``resolution`` applies to the trimesh / pyrender viewers but not
    to the viser viewer, which serves over a browser). When the factory is
    called with backend-agnostic convenience arguments, options that the
    selected backend does not understand are dropped here so the same
    ``create_viewer(name, resolution=...)`` call works for every backend.
    """
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return dict(kwargs)
    parameters = signature.parameters.values()
    # If the constructor accepts **kwargs, forward everything as-is.
    if any(p.kind == p.VAR_KEYWORD for p in parameters):
        return dict(kwargs)
    accepted = {p.name for p in parameters if p.name != 'self'}
    return {k: v for k, v in kwargs.items() if k in accepted}


def create_viewer(name='trimesh', **kwargs):
    """Create a viewer instance by backend name.

    This is the recommended entry point for examples and applications that
    want to let the user choose a viewer backend at runtime, replacing the
    ``if name == 'trimesh': ... elif name == 'pyrender': ...`` blocks that
    were previously duplicated across scripts.

    Parameters
    ----------
    name : str, optional
        Backend to instantiate. One of ``'trimesh'``, ``'pyrender'``,
        ``'viser'`` or ``'notebook'``. Default is ``'trimesh'``.
    **kwargs
        Keyword arguments forwarded to the selected viewer's constructor
        (e.g. ``resolution``, ``update_interval``, ``title`` for the
        trimesh / pyrender viewers, or ``enable_ik`` for the viser viewer).
        Arguments that the selected backend does not accept are ignored so
        the same call site can target any backend.

    Returns
    -------
    viewer : object
        An instance of the requested viewer class.

    Raises
    ------
    ValueError
        If ``name`` is not a known viewer backend.
    """
    if not isinstance(name, str):
        raise ValueError(
            "Viewer name must be a string, got {}.".format(type(name).__name__))
    key = name.lower()
    if key not in _VIEWER_CLASSES:
        raise ValueError(
            "Unknown viewer '{}'. Available viewers: {}.".format(
                name, ', '.join(sorted(_VIEWER_CLASSES))))
    cls = _VIEWER_CLASSES[key]
    return cls(**_supported_kwargs(cls, kwargs))
