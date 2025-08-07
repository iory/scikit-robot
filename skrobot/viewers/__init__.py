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

try:
    from ._trimesh import TrimeshSceneViewer
except TypeError:
    # trimesh.viewer.SceneViewer can have function type.
    class TrimeshSceneViewer(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('TrimeshSceneViewer cannot be initialized. '
                               'This issue happens when the X window system '
                               'is not running.')
except ImportError as error_log:
    warn_gl(error_log)
    class TrimeshSceneViewer(DummyViewer):
        pass

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
except ImportError as error_log:
    warn_gl(error_log)
    class PyrenderViewer(DummyViewer):
        pass
