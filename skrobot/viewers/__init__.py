# flake8: noqa

try:
    from ._trimesh import TrimeshSceneViewer
except TypeError:
    # trimesh.viewer.SceneViewer can have function type.
    class TrimeshSceneViewer(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('TrimeshSceneViewer cannot be initialized. '
                               'This issue happens when the X window system '
                               'is not running.')
