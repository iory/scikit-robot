from __future__ import division

import collections
import logging
import threading
import warnings

import numpy as np
import pyglet
from pyglet import compat_platform
import trimesh
from trimesh.transformations import euler_from_matrix
import trimesh.viewer
from trimesh.viewer.trackball import Trackball

from .. import model as model_module


logger = logging.getLogger('trimesh')
logger.setLevel(logging.ERROR)


def _check_trimesh_version():
    trimesh_version = tuple(map(int, trimesh.__version__.split('.')))
    if (4, 6, 1) < trimesh_version < (4, 6, 6):
        warnings.warn(
            "\033[31m"
            + "Trimesh version {} detected. ".format(trimesh.__version__)
            + "Versions >= 4.6.1 and <= 4.6.5 may cause models to "
            + "appear completely black."
            + "\033[0m",
            category=RuntimeWarning
        )


def _redraw_all_windows():
    for window in pyglet.app.windows:
        window.switch_to()
        window.dispatch_events()
        window.dispatch_event('on_draw')
        window.flip()
        window._legacy_invalid = False


class TrimeshSceneViewer(trimesh.viewer.SceneViewer):
    """TrimeshSceneViewer class implemented as a Singleton.

    This ensures that only one instance of the viewer
    is created throughout the program. Any subsequent attempts to create a new
    instance will return the existing one.

    Parameters
    ----------
    resolution : tuple, optional
        The resolution of the viewer. Default is (640, 480).
    update_interval : float, optional
        The update interval (in seconds) for the viewer. Default is
        1.0 seconds.
    title : str, optional
        The title of the viewer window. Default is 'scikit-robot TrimeshSceneViewer'.

    Notes
    -----
    Since this is a singleton, the __init__ method might be called
    multiple times, but only one instance is actually used.
    """

    # Class variable to hold the single instance of the class.
    _instance = None
    _version_warning_issued = False

    def __init__(self, resolution=None, update_interval=1.0, title=None):
        if getattr(self, '_initialized', False):
            return
        if resolution is None:
            resolution = (640, 480)

        self.thread = None

        self._links = collections.OrderedDict()

        self._redraw = True
        self._is_active = False
        pyglet.clock.schedule_interval(self.on_update, update_interval)

        self.scene = trimesh.Scene()
        caption = title if title is not None else 'scikit-robot TrimeshSceneViewer'
        self._kwargs = dict(
            scene=self.scene,
            resolution=resolution,
            offset_lines=False,
            start_loop=False,
            caption=caption,
        )

        self.lock = threading.Lock()
        self._initialized = True

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TrimeshSceneViewer, cls).__new__(cls)

            if not cls._version_warning_issued:
                _check_trimesh_version()
                cls._version_warning_issued = True

        return cls._instance

    def show(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.set_camera([np.deg2rad(45), -np.deg2rad(0), np.deg2rad(135)])
        if compat_platform == 'darwin':
            super(TrimeshSceneViewer, self).__init__(**self._kwargs)
            self._is_active = True
            init_loop = 30
            for _ in range(init_loop):
                _redraw_all_windows()
        else:
            self._is_active = True
            self.thread = threading.Thread(target=self._init_and_start_app)
            self.thread.daemon = True  # terminate when main thread exit
            self.thread.start()

    def _init_and_start_app(self):
        with self.lock:
            try:
                super(TrimeshSceneViewer, self).__init__(**self._kwargs)
                self._is_active = True
            except pyglet.canvas.xlib.NoSuchDisplayException:
                print('No display found. Viewer is disabled.')
                self.has_exit = True
                self._is_active = False
                return
        pyglet.app.run()
        self._is_active = False

    def redraw(self):
        self._redraw = True
        if compat_platform == 'darwin':
            _redraw_all_windows()

    def on_update(self, dt):
        self.on_draw()

    def on_draw(self):
        if not self._redraw:
            with self.lock:
                self._update_vertex_list()
                super(TrimeshSceneViewer, self).on_draw()
            return

        with self.lock:
            self._update_vertex_list()

            # apply latest angle-vector
            for link_id, link in self._links.items():
                link.update(force=True)
                transform = link.worldcoords().T()
                self.scene.graph.update(link_id, matrix=transform)
            super(TrimeshSceneViewer, self).on_draw()

        self._redraw = False

    def on_mouse_press(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_mouse_press(*args, **kwargs)

    def on_mouse_drag(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_mouse_drag(*args, **kwargs)

    def on_mouse_scroll(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_mouse_scroll(*args, **kwargs)

    def on_key_press(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_key_press(*args, **kwargs)

    def on_resize(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_resize(*args, **kwargs)

    def _add_link(self, link):
        assert isinstance(link, model_module.Link)

        with self.lock:
            link_id = str(id(link))
            if link_id in self._links:
                return
            transform = link.worldcoords().T()
            mesh = link.concatenated_visual_mesh
            # TODO(someone) fix this at trimesh's scene.
            if (isinstance(mesh, list) or isinstance(mesh, tuple)) \
               and len(mesh) > 0:
                for m in mesh:
                    link_mesh_id = link_id + str(id(m))
                    self.scene.add_geometry(
                        geometry=m,
                        node_name=link_mesh_id,
                        geom_name=link_mesh_id,
                        transform=transform,
                    )
                    self._links[link_mesh_id] = link
            elif mesh is not None:
                self.scene.add_geometry(
                    geometry=mesh,
                    node_name=link_id,
                    geom_name=link_id,
                    transform=transform,
                )
                self._links[link_id] = link

        for child_link in link._child_links:
            self._add_link(child_link)

    def add(self, geometry):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        for link in links:
            self._add_link(link)

        self._redraw = True

    def delete(self, geometry):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        with self.lock:
            for link in links:
                link_id = str(id(link))
                if link_id not in self._links:
                    continue
                self.scene.delete_geometry(link_id)
                self._links.pop(link_id)
            self.cleanup_geometries()

        self._redraw = True

    def set_camera(self, *args, **kwargs):
        if len(args) < 1 and 'angles' not in kwargs:
            if hasattr(self, "view"):
                kwargs['angles'] = euler_from_matrix(
                    self.view["ball"].pose[:3, :3])
        with self.lock:
            self.scene.set_camera(*args, **kwargs)
            if hasattr(self, "view"):
                self.view["ball"] = Trackball(
                    pose=self.scene.camera_transform,
                    size=self.scene.camera.resolution,
                    scale=self.scene.scale,
                    target=self.scene.centroid
                )

    def save_image(self, file_obj):
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
        return super(TrimeshSceneViewer, self).save_image(file_obj)

    def on_close(self):
        """Handle window close event by setting is_active to False."""
        self._is_active = False
        super(TrimeshSceneViewer, self).on_close()

    @property
    def is_active(self):
        """bool : `True` if the viewer is active, or `False` if it has been closed."""
        return self._is_active
