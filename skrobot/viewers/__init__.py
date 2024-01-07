# flake8: noqa

import inspect


try:
    from ._trimesh import TrimeshSceneViewer
except TypeError:
    # trimesh.viewer.SceneViewer can have function type.
    class TrimeshSceneViewer(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('TrimeshSceneViewer cannot be initialized. '
                               'This issue happens when the X window system '
                               'is not running.')

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
                    'pip uninstall -y pyrender && pip install git+https://github.com/mmatl/pyrender.git --no-cache-dir'
                )
except ImportError:
    class PyrenderViewer(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('PyrenderViewer is not installed.')
