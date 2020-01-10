import threading

import pyglet
import trimesh.viewer

from .. import model as model_module


class TrimeshSceneViewer(trimesh.viewer.SceneViewer):

    def __init__(self, resolution=None):
        self._links = []

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
        self.thread = threading.Thread(target=self._init_and_start_app)
        self.thread.daemon = True  # terminate when main thread exit
        self.thread.start()

    def _init_and_start_app(self):
        with self.lock:
            super(TrimeshSceneViewer, self).__init__(**self._kwargs)
        pyglet.app.run()

    def redraw(self):
        self._redraw = True

    def on_update(self, dt):
        self.on_draw()

    def on_draw(self):
        if not self._redraw:
            super(TrimeshSceneViewer, self).on_draw()
            return

        with self.lock:
            self._update_vertex_list()

            # apply latest angle-vector
            for link in self._links:
                if isinstance(link, model_module.Link):
                    link_list = [link]
                else:
                    link_list = link.link_list
                for l in link_list:
                    transform = l.worldcoords().T()
                    name = '{}/{}'.format(
                        link.__class__.__name__,
                        l.name,
                    )
                    self.scene.graph.update(name, matrix=transform)
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

    def add(self, link):
        if isinstance(link, model_module.Link):
            link_list = [link]
        elif isinstance(link, model_module.CascadedLink):
            link_list = link.link_list
        else:
            raise TypeError('link must be Link or CascadedLink')

        with self.lock:
            for l in link_list:
                transform = l.worldcoords().T()
                name = '{}/{}'.format(
                    link.__class__.__name__,
                    l.name,
                )
                self.scene.add_geometry(
                    geometry=l.visual_mesh,
                    node_name=name,
                    geom_name=name,
                    transform=transform,
                )
            self._links.append(link)

        self._redraw = True

    def set_camera(self, *args, **kwargs):
        with self.lock:
            self.scene.set_camera(*args, **kwargs)
