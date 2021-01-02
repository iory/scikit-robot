from __future__ import division

import collections
import logging
import threading

import pyglet
from pyglet import compat_platform
import trimesh
import trimesh.viewer

from .. import model as model_module


logger = logging.getLogger('trimesh')
logger.setLevel(logging.ERROR)


def _redraw_all_windows():
    for window in pyglet.app.windows:
        window.switch_to()
        window.dispatch_events()
        window.dispatch_event('on_draw')
        window.flip()
        window._legacy_invalid = False


class TrimeshSceneViewer(trimesh.viewer.SceneViewer):

    def __init__(self, resolution=None):
        if resolution is None:
            resolution = (640, 480)

        self._links = collections.OrderedDict()

        self._redraw = True
        pyglet.clock.schedule_interval(self.on_update, 1 / 30)

        self.scene = trimesh.Scene()
        self._kwargs = dict(
            scene=self.scene,
            resolution=resolution,
            offset_lines=False,
            start_loop=False,
        )

        self.lock = threading.Lock()

    def show(self):
        if compat_platform == 'darwin':
            super(TrimeshSceneViewer, self).__init__(**self._kwargs)
            init_loop = 30
            for _ in range(init_loop):
                _redraw_all_windows()
        else:
            self.thread = threading.Thread(target=self._init_and_start_app)
            self.thread.daemon = True  # terminate when main thread exit
            self.thread.start()

    def _init_and_start_app(self):
        with self.lock:
            super(TrimeshSceneViewer, self).__init__(**self._kwargs)
        pyglet.app.run()

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
            mesh = link.visual_mesh
            # TODO(someone) fix this at trimesh's scene.
            if (isinstance(mesh, list) or isinstance(mesh, tuple)) \
               and len(mesh) > 0:
                mesh = trimesh.util.concatenate(mesh)
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
        with self.lock:
            self.scene.set_camera(*args, **kwargs)
